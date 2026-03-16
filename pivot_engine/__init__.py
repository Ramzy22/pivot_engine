"""
pivot_engine package - enhanced with scalable capabilities

Expose controllers for both basic and scalable pivot operations.
"""
from .controller import PivotController
from .scalable_pivot_controller import ScalablePivotController
from .tanstack_adapter import create_tanstack_adapter, TanStackRequest, TanStackOperation, TanStackPivotAdapter
from .pivot_engine.dash_integration import register_pivot_app
from .pivot_engine.runtime import (
    DashPivotInstanceConfig,
    PivotRequestContext,
    PivotRuntimeService,
    PivotServiceResponse,
    PivotViewState,
    SessionRequestGate,
    register_dash_callbacks_for_instances,
    register_dash_drill_modal_callback,
    register_dash_pivot_transport_callback,
)

__all__ = [
    "PivotController",
    "ScalablePivotController",
    "create_tanstack_adapter",
    "TanStackRequest",
    "TanStackOperation",
    "TanStackPivotAdapter",
    "register_pivot_app",
    "DashPivotInstanceConfig",
    "PivotRequestContext",
    "PivotRuntimeService",
    "PivotServiceResponse",
    "PivotViewState",
    "SessionRequestGate",
    "register_dash_callbacks_for_instances",
    "register_dash_drill_modal_callback",
    "register_dash_pivot_transport_callback",
]
