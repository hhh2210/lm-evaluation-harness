from __future__ import annotations


"""lm_eval.registry — **lazy‑aware** refactor (draft)

One generic :class:`Registry` plus a small :class:`MetricSpec` dataclass replace
all ad‑hoc ``*_REGISTRY`` dictionaries.  New in this revision:

* **Lazy placeholders** (`"pkg.mod:Obj"` strings or *EntryPoint* objects)
  are stored until the first ``get()`` call, deferring heavy imports.
* Same decorator / function API; just add ``lazy="pkg.mod:Obj"`` or pass an
  *EntryPoint* instead of a class/function.
* Thread‑safe, type‑checked, backwards‑compatible stubs for the legacy
  globals (``MODEL_REGISTRY``, etc.).
* ``freeze_all()`` leaves first‑use materialisation intact but blocks new
  registrations, preserving determinism after bootstrap.
"""

import importlib
import inspect
import threading
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Type,
    TypeVar,
    Union,
)


try:  # Python≥3.10
    import importlib.metadata as md  # noqa: N812 (we keep the alias short)
except ImportError:  # pragma: no cover -fallback for 3.8/3.9 runtimes
    import importlib_metadata as md  # type: ignore

__all__ = [
    "Registry",
    "MetricSpec",
    # concrete registries
    "model_registry",
    "task_registry",
    "metric_registry",
    "metric_agg_registry",
    "higher_is_better_registry",
    "filter_registry",
    # helper
    "freeze_all",
    # Legacy compatibility
    "DEFAULT_METRIC_REGISTRY",
    "AGGREGATION_REGISTRY",
    "register_model",
    "get_model",
    "register_task",
    "get_task",
    "register_metric",
    "get_metric",
    "register_metric_aggregation",
    "get_metric_aggregation",
    "register_higher_is_better",
    "is_higher_better",
    "register_filter",
    "get_filter",
    "register_aggregation",
    "get_aggregation",
    "MODEL_REGISTRY",
    "TASK_REGISTRY",
    "METRIC_REGISTRY",
    "METRIC_AGGREGATION_REGISTRY",
    "HIGHER_IS_BETTER_REGISTRY",
    "FILTER_REGISTRY",
]

T = TypeVar("T")


# ────────────────────────────────────────────────────────────────────────
# Generic Registry
# ────────────────────────────────────────────────────────────────────────


class Registry(Generic[T]):
    """Name→object mapping with decorator helpers and **lazy import** support."""

    #: The underlying mutable mapping (might turn into MappingProxy on freeze)
    _objects: MutableMapping[str, Union[T, str, md.EntryPoint]]

    def __init__(
        self,
        name: str,
        *,
        base_cls: Type[T] | None = None,
        store: MutableMapping[str, Union[T, str, md.EntryPoint]] | None = None,
        validator: Callable[[T], bool] | None = None,
    ) -> None:
        self._name: str = name
        self._base_cls: Type[T] | None = base_cls
        self._objects = store if store is not None else {}
        self._metadata: Dict[
            str, Dict[str, Any]
        ] = {}  # Store metadata for each registered item
        self._validator = validator  # Custom validation function
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration helpers (decorator or direct call)
    # ------------------------------------------------------------------

    def register(
        self,
        *aliases: str,
        lazy: str | md.EntryPoint | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Callable[[T], T]:
        """``@registry.register("foo")`` **or** ``registry.register("foo", lazy="a.b:C")``.

        * If called as a **decorator**, supply an object and *no* ``lazy``.
        * If called as a **plain function** and you want lazy import, leave the
          object out and pass ``lazy=``.
        """

        def _do_register(target: Union[T, str, md.EntryPoint]) -> None:
            if not aliases:
                _aliases = (getattr(target, "__name__", str(target)),)
            else:
                _aliases = aliases

            with self._lock:
                for alias in _aliases:
                    if alias in self._objects:
                        # If it's a lazy placeholder being replaced by the concrete object, allow it
                        existing = self._objects[alias]
                        if isinstance(existing, (str, md.EntryPoint)) and isinstance(
                            target, type
                        ):
                            # Allow replacing lazy placeholder with concrete class
                            pass
                        else:
                            raise ValueError(
                                f"{self._name!r} '{alias}' already registered "
                                f"({self._objects[alias]})"
                            )
                    # Eager type check only when we have a concrete class
                    if self._base_cls is not None and isinstance(target, type):
                        if not issubclass(target, self._base_cls):  # type: ignore[arg-type]
                            raise TypeError(
                                f"{target} must inherit from {self._base_cls} "
                                f"to be registered as a {self._name}"
                            )
                    self._objects[alias] = target
                    # Store metadata if provided
                    if metadata:
                        self._metadata[alias] = metadata

        # ─── decorator path ───
        def decorator(obj: T) -> T:  # type: ignore[valid-type]
            _do_register(obj)
            return obj

        # ─── direct‑call path with lazy placeholder ───
        if lazy is not None:
            _do_register(lazy)
            return lambda x: x  # no‑op decorator for accidental use

        return decorator

    def register_bulk(
        self,
        items: Dict[str, Union[T, str, md.EntryPoint]],
        metadata: Dict[str, Dict[str, Any]] | None = None,
    ) -> None:
        """Register multiple items at once.

        Args:
            items: Dictionary mapping aliases to objects/lazy paths
            metadata: Optional dictionary mapping aliases to metadata
        """
        with self._lock:
            for alias, target in items.items():
                if alias in self._objects:
                    # If it's a lazy placeholder being replaced by the concrete object, allow it
                    existing = self._objects[alias]
                    if isinstance(existing, (str, md.EntryPoint)) and isinstance(
                        target, type
                    ):
                        # Allow replacing lazy placeholder with concrete class
                        pass
                    else:
                        raise ValueError(
                            f"{self._name!r} '{alias}' already registered "
                            f"({self._objects[alias]})"
                        )

                # Eager type check only when we have a concrete class
                if self._base_cls is not None and isinstance(target, type):
                    if not issubclass(target, self._base_cls):  # type: ignore[arg-type]
                        raise TypeError(
                            f"{target} must inherit from {self._base_cls} "
                            f"to be registered as a {self._name}"
                        )

                self._objects[alias] = target

                # Store metadata if provided
                if metadata and alias in metadata:
                    self._metadata[alias] = metadata[alias]

    # ------------------------------------------------------------------
    # Lookup & materialisation
    # ------------------------------------------------------------------

    @lru_cache(maxsize=256)  # Bounded cache to prevent memory growth
    def _materialise(self, target: Union[T, str, md.EntryPoint]) -> T:  # noqa: ANN401 -dynamic return
        """Import *target* if it is a dotted‑path string or EntryPoint."""
        if isinstance(target, str):
            mod, _, obj_name = target.partition(":")
            if not _:
                raise ValueError(
                    f"Lazy path '{target}' must be in 'module:object' form"
                )
            module = importlib.import_module(mod)
            return getattr(module, obj_name)
        if isinstance(target, md.EntryPoint):
            return target.load()
        return target  # concrete already

    def get(self, alias: str) -> T:
        with self._lock:
            try:
                target = self._objects[alias]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown {self._name} '{alias}'. Available: "
                    f"{', '.join(self._objects)}"
                ) from exc

            concrete: T = self._materialise(target)

            # First‑touch: swap placeholder with concrete obj for future calls
            if concrete is not target:
                self._objects[alias] = concrete

            # Late type check (for placeholders)
            if self._base_cls is not None and not issubclass(concrete, self._base_cls):  # type: ignore[arg-type]
                raise TypeError(
                    f"{concrete} does not inherit from {self._base_cls} "
                    f"(registered under alias '{alias}')"
                )

            # Custom validation
            if self._validator is not None and not self._validator(concrete):
                raise ValueError(
                    f"{concrete} failed custom validation for {self._name} registry "
                    f"(registered under alias '{alias}')"
                )

            return concrete

    # Mapping / dunder helpers -------------------------------------------------

    def __getitem__(self, alias: str) -> T:  # noqa: DunderImplemented
        return self.get(alias)

    def __iter__(self):  # noqa: DunderImplemented
        return iter(self._objects)

    def __len__(self) -> int:  # noqa: DunderImplemented
        return len(self._objects)

    def items(self):  # noqa: DunderImplemented
        return self._objects.items()

    # Introspection -----------------------------------------------------------

    def origin(self, alias: str) -> str | None:
        obj = self._objects.get(alias)
        try:
            if isinstance(obj, str) or isinstance(obj, md.EntryPoint):
                return None  # placeholder - unknown until imported
            file = inspect.getfile(obj)  # type: ignore[arg-type]
            line = inspect.getsourcelines(obj)[1]  # type: ignore[arg-type]
            return f"{file}:{line}"
        except (
            TypeError,
            OSError,
            AttributeError,
        ):  # pragma: no cover - best-effort only
            # TypeError: object not suitable for inspect
            # OSError: file not found or accessible
            # AttributeError: object lacks expected attributes
            return None

    def get_metadata(self, alias: str) -> Dict[str, Any] | None:
        """Get metadata for a registered item."""
        with self._lock:
            return self._metadata.get(alias)

    # Mutability --------------------------------------------------------------

    def freeze(self):
        """Make the registry *names* immutable (materialisation still works)."""
        with self._lock:
            if isinstance(self._objects, MappingProxyType):
                return  # already frozen
            self._objects = MappingProxyType(dict(self._objects))  # type: ignore[assignment]

    def clear(self):
        """Clear the registry (useful for tests). Cannot be called on frozen registries."""
        with self._lock:
            if isinstance(self._objects, MappingProxyType):
                raise RuntimeError("Cannot clear a frozen registry")
            self._objects.clear()
            self._metadata.clear()
            self._materialise.cache_clear()  # type: ignore[attr-defined] # Added by lru_cache


# ────────────────────────────────────────────────────────────────────────
# Structured objects stored in registries
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricSpec:
    """Bundle compute fn, aggregator, and *higher‑is‑better* flag."""

    compute: Callable[[Any, Any], Any]
    aggregate: Callable[[Iterable[Any]], Mapping[str, float]]
    higher_is_better: bool = True
    output_type: Optional[str] = None  # e.g., "probability", "string", "numeric"
    requires: Optional[List[str]] = None  # Dependencies on other metrics/data


# ────────────────────────────────────────────────────────────────────────
# Concrete registries used by lm_eval
# ────────────────────────────────────────────────────────────────────────

from lm_eval.api.model import LM  # noqa: E402 (local import by design)


model_registry: Registry[Type[LM]] = Registry("model", base_cls=LM)
task_registry: Registry[Callable[..., Any]] = Registry("task")
metric_registry: Registry[MetricSpec] = Registry("metric")
metric_agg_registry: Registry[Callable[[Iterable[Any]], Mapping[str, float]]] = (
    Registry("metric aggregation")
)
higher_is_better_registry: Registry[bool] = Registry("higher‑is‑better flag")
filter_registry: Registry[Callable] = Registry("filter")

# Default metric registry for output types
DEFAULT_METRIC_REGISTRY = {
    "loglikelihood": [
        "perplexity",
        "acc",
    ],
    "loglikelihood_rolling": ["word_perplexity", "byte_perplexity", "bits_per_byte"],
    "multiple_choice": ["acc", "acc_norm"],
    "generate_until": ["exact_match"],
}

# Aggregation registry (will be populated by register_aggregation)
AGGREGATION_REGISTRY: Dict[str, Callable] = {}

# ────────────────────────────────────────────────────────────────────────
# Public helper aliases (legacy API)
# ────────────────────────────────────────────────────────────────────────

register_model = model_registry.register
get_model = model_registry.get

register_task = task_registry.register
get_task = task_registry.get


# Special handling for metric registration which uses different API
def register_metric(**kwargs):
    """Register a metric with metadata.

    Compatible with old registry API that used keyword arguments.
    """

    def decorate(fn):
        metric_name = kwargs.get("metric")
        if not metric_name:
            raise ValueError("metric name is required")

        # Create MetricSpec with the function and metadata
        spec = MetricSpec(
            compute=fn,
            aggregate=lambda x: {},  # Default aggregation returns empty dict
            higher_is_better=kwargs.get("higher_is_better", True),
            output_type=kwargs.get("output_type"),
            requires=kwargs.get("requires"),
        )

        # Register in metric registry
        metric_registry._objects[metric_name] = spec

        # Also handle aggregation if specified
        if "aggregation" in kwargs:
            agg_name = kwargs["aggregation"]
            if agg_name in metric_agg_registry._objects:
                spec = MetricSpec(
                    compute=fn,
                    aggregate=metric_agg_registry._objects[agg_name],
                    higher_is_better=kwargs.get("higher_is_better", True),
                    output_type=kwargs.get("output_type"),
                    requires=kwargs.get("requires"),
                )
                metric_registry._objects[metric_name] = spec

        # Handle higher_is_better registry
        if "higher_is_better" in kwargs:
            higher_is_better_registry._objects[metric_name] = kwargs["higher_is_better"]

        return fn

    return decorate


def get_metric(name: str, hf_evaluate_metric=False):
    """Get a metric by name, with fallback to HF evaluate."""
    if not hf_evaluate_metric:
        try:
            spec = metric_registry.get(name)
            if isinstance(spec, MetricSpec):
                return spec.compute
            return spec
        except KeyError:
            import logging

            logging.getLogger(__name__).warning(
                f"Could not find registered metric '{name}' in lm-eval, searching in HF Evaluate library..."
            )

    # Fallback to HF evaluate
    try:
        import evaluate as hf_evaluate

        metric_object = hf_evaluate.load(name)
        return metric_object.compute
    except Exception:
        import logging

        logging.getLogger(__name__).error(
            f"{name} not found in the evaluate library! Please check https://huggingface.co/evaluate-metric",
        )
        return None


register_metric_aggregation = metric_agg_registry.register
get_metric_aggregation = metric_agg_registry.get

register_higher_is_better = higher_is_better_registry.register
is_higher_better = higher_is_better_registry.get

register_filter = filter_registry.register
get_filter = filter_registry.get


# Special handling for AGGREGATION_REGISTRY which works differently
def register_aggregation(name: str):
    def decorate(fn):
        if name in AGGREGATION_REGISTRY:
            raise ValueError(
                f"aggregation named '{name}' conflicts with existing registered aggregation!"
            )
        AGGREGATION_REGISTRY[name] = fn
        return fn

    return decorate


def get_aggregation(name: str) -> Callable[[], Dict[str, Callable]]:
    try:
        return AGGREGATION_REGISTRY[name]
    except KeyError:
        import logging

        logging.getLogger(__name__).warning(
            f"{name} not a registered aggregation metric!"
        )
        return None


# ────────────────────────────────────────────────────────────────────────
# Optional PyPI entry‑point discovery - uncomment if desired
# ────────────────────────────────────────────────────────────────────────

# for _group, _reg in {
#     "lm_eval.models": model_registry,
#     "lm_eval.tasks": task_registry,
#     "lm_eval.metrics": metric_registry,
# }.items():
#     for _ep in md.entry_points(group=_group):
#         _reg.register(_ep.name, lazy=_ep)


# ────────────────────────────────────────────────────────────────────────
# Convenience
# ────────────────────────────────────────────────────────────────────────


def freeze_all() -> None:  # pragma: no cover
    """Freeze every global registry (idempotent)."""
    for _reg in (
        model_registry,
        task_registry,
        metric_registry,
        metric_agg_registry,
        higher_is_better_registry,
        filter_registry,
    ):
        _reg.freeze()


# ────────────────────────────────────────────────────────────────────────
# Backwards‑compatibility read‑only globals
# ────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: Mapping[str, Type[LM]] = MappingProxyType(model_registry._objects)  # type: ignore[attr-defined]
TASK_REGISTRY: Mapping[str, Callable[..., Any]] = MappingProxyType(
    task_registry._objects
)  # type: ignore[attr-defined]
METRIC_REGISTRY: Mapping[str, MetricSpec] = MappingProxyType(metric_registry._objects)  # type: ignore[attr-defined]
METRIC_AGGREGATION_REGISTRY: Mapping[str, Callable] = MappingProxyType(
    metric_agg_registry._objects
)  # type: ignore[attr-defined]
HIGHER_IS_BETTER_REGISTRY: Mapping[str, bool] = MappingProxyType(
    higher_is_better_registry._objects
)  # type: ignore[attr-defined]
FILTER_REGISTRY: Mapping[str, Callable] = MappingProxyType(filter_registry._objects)  # type: ignore[attr-defined]
