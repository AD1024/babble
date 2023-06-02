//! Extracting expressions with learned libs out of egraphs

pub mod beam;

#[cfg(feature = "grb")]
pub mod ilp;

pub mod ilp_extract;
pub mod maxsat_extract;

use std::collections::HashMap;

use egg::{Analysis, EGraph, Id, Language, RecExpr, Rewrite, Runner};
use rplex;
use time::Instant;

use crate::{
    ast_node::{Arity, AstNode, Expr},
    learn::LibId,
    teachable::{BindingExpr, Teachable},
};

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum ExtractorType {
    ILPACYC,
    ILPTOPO,
    MAXSAT,
    DEFAULT,
}

pub struct LpAstSize;
impl<Op, A> maxsat_extract::MaxsatCostFunction<AstNode<Op>, A> for LpAstSize
where
    Op: Clone
        + Teachable
        + Ord
        + std::fmt::Debug
        + std::fmt::Display
        + std::hash::Hash
        + Arity
        + Send
        + Sync,
    A: Analysis<AstNode<Op>> + Default + Clone,
{
    fn node_cost(
        &mut self,
        egraph: &EGraph<AstNode<Op>, A>,
        eclass: Id,
        enode: &AstNode<Op>,
    ) -> f64 {
        1.0
    }
}

/// Given an `egraph` that contains the original expression at `roots`,
/// and a set of library `rewrites`, extract the programs rewritten using the library.
pub fn apply_libs<Op, A>(
    egraph: EGraph<AstNode<Op>, A>,
    roots: &[Id],
    rewrites: &[Rewrite<AstNode<Op>, A>],
    extractor_type: ExtractorType,
) -> RecExpr<AstNode<Op>>
where
    Op: Clone
        + Teachable
        + Ord
        + std::fmt::Debug
        + std::fmt::Display
        + std::hash::Hash
        + Arity
        + Send
        + Sync,
    A: Analysis<AstNode<Op>> + Default + Clone,
{
    let mut fin = Runner::<_, _, ()>::new(Default::default())
        .with_egraph(egraph)
        .run(rewrites.iter())
        .egraph;
    let root = fin.add(AstNode::new(Op::list(), roots.iter().copied()));

    if extractor_type == ExtractorType::MAXSAT {
        let mut maxsat_ext = maxsat_extract::MaxsatExtractor::new(&fin, "apply_lib.p".into());
        let problem = maxsat_ext.create_problem(root, "lib_ext", true, LpAstSize);
        let (elapsed, cost, best): (u128, Option<f64>, RecExpr<AstNode<Op>>) =
            problem.solve().unwrap();
        println!("WPMAXSAT Extract Time: {}", elapsed);
        lift_libs(best)
    } else if extractor_type == ExtractorType::ILPACYC || extractor_type == ExtractorType::ILPTOPO {
        let env = rplex::Env::new().unwrap();
        let mut ilp_problem = ilp_extract::create_problem(
            &env,
            root,
            &fin,
            true,
            extractor_type == ExtractorType::ILPTOPO,
            |_, _, _| 1.0,
        );
        let (elapsed, opt, best) = ilp_problem.solve();
        println!(
            "ILP-{} Extract Time: {}",
            if extractor_type == ExtractorType::ILPACYC {
                "ACYC"
            } else {
                "TOPO"
            },
            elapsed
        );
        lift_libs(best)
    } else {
        let start = Instant::now();
        let mut extractor = beam::LibExtractor::new(&fin);
        let best = extractor.best(root);
        let end = Instant::now();
        lift_libs(best)
    }
}

/// Lifts libs
pub fn lift_libs<Op>(expr: RecExpr<AstNode<Op>>) -> RecExpr<AstNode<Op>>
where
    Op: Clone + Teachable + Ord + std::fmt::Debug + std::hash::Hash,
{
    let orig: Vec<AstNode<Op>> = expr.as_ref().to_vec();
    let mut seen = HashMap::new();

    fn build<Op: Clone + Teachable + std::fmt::Debug>(
        orig: &[AstNode<Op>],
        cur: Id,
        mut seen: impl FnMut(LibId, Id),
    ) -> AstNode<Op> {
        match orig[Into::<usize>::into(cur)].as_binding_expr() {
            Some(BindingExpr::Lib(id, lam, c)) => {
                seen(id, *lam);
                build(orig, *c, seen)
            }
            _ => orig[Into::<usize>::into(cur)].clone(),
        }
    }

    let rest = orig[orig.len() - 1].build_recexpr(|id| {
        build(&orig, id, |k, v| {
            seen.insert(k, v);
        })
    });
    let mut res = rest.as_ref().to_vec();

    // Work queue for functions we still have to do
    let mut q: Vec<(LibId, Id)> = seen.iter().map(|(k, v)| (*k, *v)).collect();

    // TODO: order based on libs dependency w each other?
    while let Some((lib, expr)) = q.pop() {
        let body = res.len() - 1;
        let value: Vec<_> = orig[Into::<usize>::into(expr)]
            .build_recexpr(|id| {
                build(&orig, id, |k, v| {
                    if let None = seen.insert(k, v) {
                        q.push((k, v));
                    }
                })
            })
            .as_ref()
            .iter()
            .cloned()
            .map(|x| x.map_children(|x| (usize::from(x) + res.len()).into()))
            .collect();
        res.extend(value);
        res.push(Teachable::lib(lib, Id::from(res.len() - 1), Id::from(body)));
    }

    res.into()
}

/// Get the true cost of an expr
pub fn true_cost<Op: Clone>(expr: RecExpr<AstNode<Op>>) -> usize {
    Expr::len(&expr.into())
}
