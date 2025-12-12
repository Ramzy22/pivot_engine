"""
ui_proxy_service.py - Microservice for UI proxy and real-time updates
"""
import asyncio
import json
from typing import Dict, Any, Optional, Callable, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import pyarrow as pa
from ...types.pivot_spec import PivotSpec
from ...scalable_pivot_controller import ScalablePivotController


class UIPivotProxy:
    """UI proxy service for handling pivot requests and real-time updates"""
    
    def __init__(self, controller: ScalablePivotController, config: Optional[Dict[str, Any]] = None):
        self.controller = controller
        self.config = config or {}
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.subscription_manager = SubscriptionManager()
        self.throttle_time = config.get('throttle_time', 0.1)  # 100ms between updates
        
    async def handle_pivot_request(self, spec_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pivot request from UI"""
        try:
            # Validate and normalize spec
            spec = PivotSpec.from_dict(spec_dict)
            
            # Execute pivot
            result = self.controller.run_pivot(spec, return_format="dict")
            
            return {
                "status": "success",
                "data": result,
                "execution_time": 0,  # Would be populated with actual timing
                "cache_info": {}  # Would contain cache hit/miss info
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }
    
    async def handle_hierarchical_pivot(self, spec_dict: Dict[str, Any], 
                                      expanded_paths: List[List[str]],
                                      user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle hierarchical pivot request"""
        try:
            spec = PivotSpec.from_dict(spec_dict)
            
            # Check if pruning should be applied
            if user_preferences and user_preferences.get('enable_pruning', True):
                # Use the controller's new pruning method
                result = await self.controller.run_pruned_hierarchical_pivot(
                    spec, expanded_paths, user_preferences
                )
            else:
                # Use standard hierarchical pivot
                result = self.controller.run_hierarchical_pivot(spec_dict)
            
            return {
                "status": "success",
                "data": result,
                "expanded_paths": expanded_paths
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }
    
    async def handle_virtual_scroll(self, spec_dict: Dict[str, Any], 
                                  start_row: int, end_row: int,
                                  expanded_paths: List[List[str]]) -> Dict[str, Any]:
        """Handle virtual scrolling request"""
        try:
            spec = PivotSpec.from_dict(spec_dict)
            
            # Use the controller's virtual scrolling capability
            result = await self.controller.run_virtual_scroll_hierarchical(
                spec, start_row, end_row, expanded_paths
            )
            
            return {
                "status": "success",
                "data": result,
                "range": {"start": start_row, "end": end_row, "total": len(result) if isinstance(result, list) else 0}
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }
    
    async def subscribe_to_updates(self, websocket: WebSocket, path: str, subscription_id: str):
        """Subscribe to real-time updates for a specific path"""
        await self.subscription_manager.add_subscription(websocket, path, subscription_id)
        
        # Add to active connections
        if path not in self.active_connections:
            self.active_connections[path] = []
        self.active_connections[path].append(websocket)
    
    async def broadcast_update(self, path: str, data: Any, metadata: Optional[Dict[str, Any]] = None):
        """Broadcast update to all subscribers of a path"""
        if path in self.active_connections:
            message = {
                "type": "data_update",
                "path": path,
                "data": data,
                "timestamp": asyncio.get_event_loop().time(),
                "metadata": metadata or {}
            }
            
            disconnected = []
            for websocket in self.active_connections[path]:
                try:
                    await websocket.send_text(json.dumps(message))
                except WebSocketDisconnect:
                    disconnected.append(websocket)
            
            # Remove disconnected websockets
            for websocket in disconnected:
                self.active_connections[path].remove(websocket)
        
        # Also notify subscription manager
        await self.subscription_manager.notify_update(path, data, metadata)


class SubscriptionManager:
    """Manage WebSocket subscriptions for real-time updates"""
    
    def __init__(self):
        self.subscriptions = {}  # subscription_id -> subscription_info
        self.path_subscriptions = {}  # path -> [subscription_ids]
        
    async def add_subscription(self, websocket: WebSocket, path: str, subscription_id: str):
        """Add a new subscription"""
        self.subscriptions[subscription_id] = {
            'websocket': websocket,
            'path': path,
            'created_at': asyncio.get_event_loop().time(),
            'last_active': asyncio.get_event_loop().time()
        }
        
        if path not in self.path_subscriptions:
            self.path_subscriptions[path] = []
        self.path_subscriptions[path].append(subscription_id)
    
    async def remove_subscription(self, subscription_id: str):
        """Remove a subscription"""
        if subscription_id in self.subscriptions:
            path = self.subscriptions[subscription_id]['path']
            del self.subscriptions[subscription_id]
            
            # Remove from path mapping
            if path in self.path_subscriptions:
                if subscription_id in self.path_subscriptions[path]:
                    self.path_subscriptions[path].remove(subscription_id)
    
    async def notify_update(self, path: str, data: Any, metadata: Optional[Dict[str, Any]] = None):
        """Notify all subscribers of a path about an update"""
        if path in self.path_subscriptions:
            for sub_id in self.path_subscriptions[path]:
                if sub_id in self.subscriptions:
                    websocket = self.subscriptions[sub_id]['websocket']
                    message = {
                        "type": "data_update",
                        "subscription_id": sub_id,
                        "path": path,
                        "data": data,
                        "timestamp": asyncio.get_event_loop().time(),
                        "metadata": metadata or {}
                    }
                    
                    try:
                        await websocket.send_text(json.dumps(message))
                        self.subscriptions[sub_id]['last_active'] = asyncio.get_event_loop().time()
                    except WebSocketDisconnect:
                        await self.remove_subscription(sub_id)


class UIPivotService:
    """Main UI service that integrates with FastAPI"""
    
    def __init__(self, controller: ScalablePivotController):
        self.controller = controller
        self.proxy = UIPivotProxy(controller)
        self.app = FastAPI(title="Scalable Pivot UI Service")
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        @self.app.post("/pivot")
        async def pivot_endpoint(spec: Dict[str, Any]):
            result = await self.proxy.handle_pivot_request(spec)
            return JSONResponse(content=result)
        
        @self.app.post("/hierarchical-pivot")
        async def hierarchical_pivot_endpoint(request_data: Dict[str, Any]):
            spec = request_data.get('spec', {})
            expanded_paths = request_data.get('expanded_paths', [])
            user_preferences = request_data.get('user_preferences', {})
            
            result = await self.proxy.handle_hierarchical_pivot(spec, expanded_paths, user_preferences)
            return JSONResponse(content=result)
        
        @self.app.get("/virtual-scroll")
        async def virtual_scroll_endpoint(start: int, end: int, spec: str, expanded_paths: str = "[]"):
            import json
            spec_dict = json.loads(spec)
            paths_list = json.loads(expanded_paths)
            
            result = await self.proxy.handle_virtual_scroll(spec_dict, start, end, paths_list)
            return JSONResponse(content=result)
        
        @self.app.websocket("/ws/{path}")
        async def websocket_endpoint(websocket: WebSocket, path: str):
            await websocket.accept()
            
            try:
                # Wait for subscription message
                data = await websocket.receive_text()
                msg = json.loads(data)
                
                if msg.get('type') == 'subscribe':
                    subscription_id = msg.get('subscription_id', f"sub_{path}_{hash(path)}")
                    await self.proxy.subscribe_to_updates(websocket, path, subscription_id)
                    
                    # Send acknowledgment
                    ack_msg = {
                        "type": "subscription_ack",
                        "subscription_id": subscription_id,
                        "status": "success"
                    }
                    await websocket.send_text(json.dumps(ack_msg))
                
                # Keep connection alive and handle any messages
                while True:
                    try:
                        # This would handle any client messages, if needed
                        data = await websocket.receive_text()
                        # Process client messages if needed
                    except WebSocketDisconnect:
                        break
                        
            except WebSocketDisconnect:
                pass  # Connection closed


# Example usage function
async def create_ui_service():
    """Create and return a configured UI service"""
    # Create a controller instance
    controller = ScalablePivotController(
        backend_uri=":memory:",
        enable_streaming=True,
        enable_incremental_views=True
    )
    
    # Create the UI service
    ui_service = UIPivotService(controller)
    
    return ui_service