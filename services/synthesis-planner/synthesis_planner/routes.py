"""AiZynthFinder RouteCollection → typed JSON serialization.

Converts AiZynthFinder's internal route tree objects into clean, serializable
Python dataclasses.  All AiZynthFinder-specific types are confined to this
module so the rest of the service can import only the plain dataclasses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReactantInfo:
    """One leaf node in a retrosynthesis route (a purchasable building block)."""
    smiles: str
    in_stock: bool
    stock_source: str | None = None   # name of the stock catalog that matched


@dataclass
class ReactionStep:
    """One reaction step in a route."""
    depth: int                        # distance from the target (1 = first retro step)
    smarts: str | None                # reaction SMARTS from the template
    template_id: str | None           # identifier of the matched template
    confidence: float | None          # model confidence for this template selection
    reactants: list[str] = field(default_factory=list)   # SMILES of reactant molecules
    products: list[str] = field(default_factory=list)    # SMILES of product molecules


@dataclass
class RouteCandidate:
    """One complete retrosynthesis route from target to building blocks."""
    route_id: str
    n_steps: int
    score: float                             # route-level score (higher = better)
    reactions: list[ReactionStep] = field(default_factory=list)
    leaves: list[ReactantInfo] = field(default_factory=list)
    all_in_stock: bool = False               # True iff every leaf is in stock


# ---------------------------------------------------------------------------
# Conversion helpers — all AiZynthFinder imports are lazy and confined here
# ---------------------------------------------------------------------------

def _extract_mol_smiles(mol_node: Any) -> str:
    """Extract the SMILES from an AiZynthFinder TreeMolecule node."""
    try:
        return mol_node.smiles
    except AttributeError:
        return str(mol_node)


def _extract_reaction_step(rxn_node: Any, depth: int) -> ReactionStep:
    """Convert one AiZynthFinder reaction node to a ReactionStep."""
    smarts: str | None = None
    template_id: str | None = None
    confidence: float | None = None
    reactants: list[str] = []
    products: list[str] = []

    try:
        metadata = rxn_node.metadata or {}
        smarts      = metadata.get("reaction_smarts") or metadata.get("smarts")
        template_id = str(metadata.get("template_hash") or metadata.get("id") or "")
        confidence  = float(metadata.get("classification") or metadata.get("confidence") or 0)
        reactants = [_extract_mol_smiles(m) for m in getattr(rxn_node, "reactants", []) or []]
        products  = [_extract_mol_smiles(m) for m in getattr(rxn_node, "products", []) or []]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not fully parse reaction node: %s", exc)

    return ReactionStep(
        depth=depth,
        smarts=smarts,
        template_id=template_id or None,
        confidence=confidence,
        reactants=reactants,
        products=products,
    )


def _walk_route_tree(tree_route: Any) -> tuple[list[ReactionStep], list[Any]]:
    """Extract reaction steps and leaf molecules from an AiZynthFinder ReactionTree.

    Supports both the v4.4+ ReactionTree object (with .leafs() / .reactions()
    methods) and older object trees with a recursive children structure.

    Returns (steps, leaf_mol_nodes).
    """
    steps: list[ReactionStep] = []
    leaves: list[Any] = []

    # v4.4+ ReactionTree has .leafs() and .reactions() generators
    if hasattr(tree_route, "leafs") and hasattr(tree_route, "reactions"):
        try:
            leaves = list(tree_route.leafs())
        except Exception:
            pass
        try:
            for depth, rxn in enumerate(tree_route.reactions(), start=1):
                steps.append(_extract_reaction_step(rxn, depth))
        except Exception:
            pass
        return steps, leaves

    # Fallback: recursive walk for older object trees
    def _walk(node: Any, depth: int) -> None:
        children = list(getattr(node, "children", None) or [])
        if not children:
            leaves.append(node)
            return
        for child in children:
            if hasattr(child, "metadata"):
                steps.append(_extract_reaction_step(child, depth))
                for grandchild in getattr(child, "children", []) or []:
                    _walk(grandchild, depth + 1)
            else:
                _walk(child, depth)

    _walk(tree_route, depth=1)
    return steps, leaves


def serialize_routes(
    route_collection: Any,
    stock: Any | None = None,
    max_routes: int = 10,
) -> list[RouteCandidate]:
    """Convert an AiZynthFinder RouteCollection to a list of RouteCandidate.

    Parameters
    ----------
    route_collection:
        The ``finder.routes`` object after ``finder.build_routes()`` was called.
    stock:
        Optional AiZynthFinder stock object for in-stock checks.
    max_routes:
        Maximum number of routes to serialize (best-first).

    Returns
    -------
    list[RouteCandidate]
        Sorted by score descending (best route first).
    """
    candidates: list[RouteCandidate] = []

    try:
        routes_iter = list(route_collection)[:max_routes]
    except Exception as exc:
        logger.warning("Could not iterate route collection: %s", exc)
        return []

    for idx, route in enumerate(routes_iter):
        route_id = f"route_{idx:03d}"
        score: float = 0.0

        try:
            # v4.4+: route is a dict with 'score' as a dict e.g. {'state score': 0.049}
            if isinstance(route, dict):
                score_val = route.get("score", 0)
                if isinstance(score_val, dict):
                    score = float(next(iter(score_val.values()), 0))
                else:
                    score = float(score_val or 0)
            else:
                # Older object-based routes
                scores_dict = getattr(route, "scores", None) or {}
                if scores_dict:
                    score = float(next(iter(scores_dict.values()), 0))
                else:
                    score = float(getattr(route, "score", 0) or 0)
        except Exception:
            pass

        # Try to get the route tree for step extraction
        steps: list[ReactionStep] = []
        leaf_mols: list[Any] = []
        try:
            # v4.4+: route is a dict with 'reaction_tree' key (ReactionTree object)
            if isinstance(route, dict):
                tree = route.get("reaction_tree")
            else:
                tree = getattr(route, "reaction_tree", None)
            if tree is not None:
                steps, leaf_mols = _walk_route_tree(tree)
        except Exception as exc:
            logger.debug("Route %s tree walk failed: %s", route_id, exc)

        # Build leaf info
        leaves: list[ReactantInfo] = []
        for mol in leaf_mols:
            smiles = _extract_mol_smiles(mol)
            in_stock = False
            stock_source: str | None = None
            if stock is not None:
                try:
                    in_stock = mol in stock
                    if in_stock:
                        stock_source = "configured_stock"
                except Exception:
                    pass
            leaves.append(ReactantInfo(
                smiles=smiles,
                in_stock=in_stock,
                stock_source=stock_source,
            ))

        candidates.append(RouteCandidate(
            route_id=route_id,
            n_steps=len(steps),
            score=score,
            reactions=steps,
            leaves=leaves,
            all_in_stock=all(lf.in_stock for lf in leaves) if leaves else False,
        ))

    # Sort best-first
    candidates.sort(key=lambda r: -r.score)
    return candidates


def route_to_dict(route: RouteCandidate) -> dict[str, Any]:
    """Convert a RouteCandidate to a JSON-serializable dict."""
    return {
        "route_id": route.route_id,
        "n_steps": route.n_steps,
        "score": route.score,
        "all_in_stock": route.all_in_stock,
        "reactions": [
            {
                "depth": s.depth,
                "smarts": s.smarts,
                "template_id": s.template_id,
                "confidence": s.confidence,
                "reactants": s.reactants,
                "products": s.products,
            }
            for s in route.reactions
        ],
        "leaves": [
            {
                "smiles": lf.smiles,
                "in_stock": lf.in_stock,
                "stock_source": lf.stock_source,
            }
            for lf in route.leaves
        ],
    }
