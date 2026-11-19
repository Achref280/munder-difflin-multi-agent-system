"""
Event emitter for broadcasting workflow events to listeners.
Agents emit events, and visualizers subscribe to them.
"""

from typing import List, Callable
from workflow_events import WorkflowEvent


class EventEmitter:
    """
    Central event bus for workflow events.
    
    Usage:
        emitter = EventEmitter()
        emitter.subscribe(visualizer.on_event)  # Visualizer listens
        
        emitter.emit(WorkflowEvent(...))  # Agent reports progress
    
    Key principle: Agents emit events independently of listeners.
    If no one is listening, events are still recorded.
    """
    
    def __init__(self):
        self.events: List[WorkflowEvent] = []
        self.listeners: List[Callable] = []
    
    def emit(self, event: WorkflowEvent) -> None:
        """
        Record an event and notify all subscribers.
        
        Args:
            event: WorkflowEvent to broadcast
        """
        # Store for history
        self.events.append(event)
        
        # Notify all listeners (they handle visualization independently)
        for listener in self.listeners:
            try:
                listener(event)
            except Exception as e:
                # Visualization errors don't crash the workflow
                print(f"⚠️  Visualization error: {e}")
    
    def subscribe(self, callback: Callable[[WorkflowEvent], None]) -> None:
        """
        Register a listener to be called on every event.
        
        Args:
            callback: Function to call with each WorkflowEvent
        """
        self.listeners.append(callback)
    
    def unsubscribe(self, callback: Callable) -> None:
        """Remove a listener."""
        if callback in self.listeners:
            self.listeners.remove(callback)
    
    def get_events(self, 
                   event_type: str = None,
                   request_id: str = None) -> List[WorkflowEvent]:
        """
        Retrieve recorded events, optionally filtered.
        
        Args:
            event_type: Filter by type ("extraction", "inventory", etc.)
            request_id: Filter by request
            
        Returns:
            List of matching WorkflowEvent objects
        """
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if request_id:
            events = [e for e in events if e.request_id == request_id]
        
        return events.copy()
    
    def clear(self) -> None:
        """Clear all recorded events (useful for testing)."""
        self.events.clear()


# Global singleton instance
# Import this in your agents and visualizers
emitter = EventEmitter()
