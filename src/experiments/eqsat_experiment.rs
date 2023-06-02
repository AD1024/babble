use std::{
    fmt::{self, Debug, Display, Formatter},
    hash::Hash,
    time::{Duration, Instant},
};

use egg::{AstSize, CostFunction, EGraph, Id, RecExpr, Rewrite, Runner};
use log::debug;
use serde::ser::Serialize;

use crate::{
    ast_node::{Arity, AstNode, Expr, Pretty, Printable},
    extract::{
        beam::{LibExtractor, PartialLibCost},
        ilp_extract, lift_libs, maxsat_extract, ExtractorType, LpAstSize,
    },
    teachable::Teachable,
};

use super::{CsvWriter, Experiment, ExperimentResult};

/// A BeamExperiment contains all of the information needed to run a
/// library learning experiment with the beam extractor.
#[derive(Debug)]
pub struct EqsatExperiment<Op, Extra>
where
    Op: Display + Hash + Clone + Ord + 'static,
{
    /// The domain-specific rewrites to apply
    dsrs: Vec<Rewrite<AstNode<Op>, PartialLibCost>>,
    /// Any extra data associated with this experiment
    extra_data: Extra,
    /// Extractor
    extractor: ExtractorType,
}

impl<Op, Extra> EqsatExperiment<Op, Extra>
where
    Op: Arity
        + Teachable
        + Printable
        + Debug
        + Display
        + Hash
        + Clone
        + Ord
        + Sync
        + Send
        + 'static,
{
    pub fn new<I>(dsrs: I, extra_data: Extra, extractor: ExtractorType) -> Self
    where
        I: IntoIterator<Item = Rewrite<AstNode<Op>, PartialLibCost>>,
    {
        Self {
            dsrs: dsrs.into_iter().collect(),
            extra_data,
            extractor,
        }
    }

    fn run_egraph(
        &self,
        roots: &[Id],
        egraph: EGraph<AstNode<Op>, PartialLibCost>,
    ) -> ExperimentResult<Op> {
        let start_time = Instant::now();
        let timeout = Duration::from_secs(60 * 100000);

        debug!("Running {} DSRs... ", self.dsrs.len());

        let runner = Runner::<_, _, ()>::new(PartialLibCost::empty())
            .with_egraph(egraph)
            .with_time_limit(timeout)
            .run(&self.dsrs);

        let mut fin = runner.egraph;

        debug!("Finished in {}ms", start_time.elapsed().as_millis());

        debug!("Extracting... ");
        let root = fin.add(AstNode::new(Op::list(), roots.iter().copied()));
        // fin.dot().to_pdf("egraph.pdf").unwrap();
        let ex_time = Instant::now();

        let lifted = if self.extractor == ExtractorType::MAXSAT {
            let now = Instant::now();
            let mut maxsat_ext = maxsat_extract::MaxsatExtractor::new(&fin, "apply_lib.p".into());
            let problem = maxsat_ext.create_problem(root, "lib_ext", true, LpAstSize);
            let (elapsed, cost, best): (u128, Option<f64>, RecExpr<AstNode<Op>>) =
                problem.solve().unwrap();
            println!("WPMAXSAT Extract Time: {} (solver time: {})", now.elapsed().as_millis(), elapsed);
            lift_libs(best)
        } else if self.extractor == ExtractorType::ILPACYC
            || self.extractor == ExtractorType::ILPTOPO
        {
            let env = rplex::Env::new().unwrap();
            let now = Instant::now();
            let mut ilp_problem = ilp_extract::create_problem(
                &env,
                root,
                &fin,
                true,
                self.extractor == ExtractorType::ILPTOPO,
                |_, _, _| 1.0,
            );
            let (elapsed, opt, best) = ilp_problem.solve();
            println!(
                "ILP-{} Extract Time: {} (solver time: {})",
                if self.extractor == ExtractorType::ILPACYC {
                    "ACYC"
                } else {
                    "TOPO"
                },
                now.elapsed().as_millis(),
                elapsed
            );
            lift_libs(best)
        } else {
            let mut extractor = LibExtractor::new(&fin);
            let best = extractor.best(root);
            println!("Default Extract Time: {}", ex_time.elapsed().as_millis());
            lift_libs(best)
        };
        let final_cost = AstSize.cost_rec(&lifted);
        debug!("Finished in {}ms", ex_time.elapsed().as_millis());
        debug!("final cost: {}", final_cost);
        debug!("{}", Pretty(&Expr::from(lifted.clone())));
        debug!("round time: {}ms", start_time.elapsed().as_millis());

        ExperimentResult {
            final_expr: lifted.into(),
            num_libs: self.dsrs.len(),
            rewrites: self.dsrs.clone(),
        }
    }
}

impl<Op, Extra> Experiment<Op> for EqsatExperiment<Op, Extra>
where
    Op: Teachable
        + Printable
        + Arity
        + Clone
        + Send
        + Sync
        + Debug
        + Display
        + Hash
        + Ord
        + 'static,
    Extra: Serialize + Debug + Clone,
{
    fn dsrs(&self) -> &[Rewrite<AstNode<Op>, PartialLibCost>] {
        &self.dsrs
    }

    fn run(&self, exprs: Vec<Expr<Op>>, _writer: &mut CsvWriter) -> ExperimentResult<Op> {
        // First, let's turn our list of exprs into a list of recexprs
        let recexprs: Vec<RecExpr<AstNode<Op>>> =
            exprs.clone().into_iter().map(|x| x.into()).collect();

        let mut egraph = EGraph::new(PartialLibCost::new(0, 0, 1, false));
        let roots: Vec<_> = recexprs.iter().map(|x| egraph.add_expr(x)).collect();
        egraph.rebuild();

        self.run_egraph(&roots, egraph)
    }

    fn total_rounds(&self) -> usize {
        1
    }

    fn run_multi(&self, expr_groups: Vec<Vec<Expr<Op>>>) -> ExperimentResult<Op> {
        // First, let's turn our list of exprs into a list of recexprs
        let recexpr_groups: Vec<Vec<_>> = expr_groups
            .into_iter()
            .map(|group| group.into_iter().map(RecExpr::from).collect())
            .collect();

        let mut egraph = EGraph::new(PartialLibCost::new(0, 0, 1, false));

        let roots: Vec<_> = recexpr_groups
            .into_iter()
            .map(|mut group| {
                let first_expr = group.pop().unwrap();
                let root = egraph.add_expr(&first_expr);
                for expr in group {
                    let class = egraph.add_expr(&expr);
                    egraph.union(root, class);
                }

                root
            })
            .into_iter()
            .collect();

        egraph.rebuild();

        self.run_egraph(&roots, egraph)
    }

    fn write_to_csv(
        &self,
        writer: &mut CsvWriter,
        round: usize,
        initial_cost: usize,
        final_cost: usize,
        compression: f64,
        num_libs: usize,
        time_elapsed: Duration,
    ) {
        writer
            .serialize((
                "eqsat",
                0,
                0,
                0,
                0,
                0,
                false,
                self.extra_data.clone(),
                round,
                initial_cost,
                final_cost,
                compression,
                num_libs,
                time_elapsed.as_secs_f64(),
            ))
            .unwrap();
        writer.flush().unwrap();
    }

    fn fmt_title(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "eqsat | extra {:?}", self.extra_data)
    }
}
