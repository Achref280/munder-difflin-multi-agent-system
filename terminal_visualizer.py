"""
Terminal visualizer for workflow events using Rich.
Displays real-time progress without affecting business logic.
"""

from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from workflow_events import WorkflowEvent


class TerminalVisualizer:
    """
    Renders workflow events beautifully in the terminal.
    
    Features:
    - Real-time status table showing current stage of each workflow phase
    - Event history showing last N events
    - Per-request tracking
    - Non-blocking (doesn't affect agent execution)
    
    Usage:
        visualizer = TerminalVisualizer()
        emitter.subscribe(visualizer.on_event)
    """
    
    # Stage emojis
    STAGE_EMOJI = {
        "started": "▶️ ",
        "in_progress": "⏳",
        "completed": "✅",
        "failed": "❌"
    }
    
    # Event type colors
    EVENT_COLORS = {
        "extraction": "cyan",
        "inventory": "blue",
        "pricing": "magenta",
        "fulfillment": "green"
    }
    
    def __init__(self, max_history: int = 10, enable_clear: bool = False):
        """
        Initialize the visualizer.
        
        Args:
            max_history: Number of recent events to display
            enable_clear: Whether to clear screen on each update
        """
        self.console = Console()
        self.max_history = max_history
        self.enable_clear = enable_clear
        
        # Track events by type
        self.event_history: dict[str, list[WorkflowEvent]] = defaultdict(list)
        
        # Track latest event per request
        self.latest_by_request: dict[str, WorkflowEvent] = {}
        
        # Track request status
        self.request_status: dict[str, str] = {}
    
    def on_event(self, event: WorkflowEvent) -> None:
        """
        Called by EventEmitter whenever an event occurs.
        This runs independently of agent logic.
        
        Args:
            event: The WorkflowEvent that occurred
        """
        # Store event
        self.event_history[event.event_type].append(event)
        self.latest_by_request[event.request_id] = event
        
        # Update request status
        self._update_request_status(event.request_id, event)
        
        # Render updated display
        self.render()
    
    def render(self) -> None:
        """Draw the current status display."""
        if self.enable_clear:
            self.console.clear()
        
        # Title
        title_text = Text("🏢 Munder Difflin Order Processing", style="bold cyan")
        self.console.print(Panel(title_text, style="cyan", expand=False))
        
        # Stage status table
        self._render_stage_status()
        
        # Recent events list
        self._render_recent_events()
        
        # Request summary
        self._render_request_summary()
    
    def _render_stage_status(self) -> None:
        """Display current status of each workflow stage."""
        self.console.print()
        table = Table(title="📊 Workflow Stages", show_header=True)
        table.add_column("Stage", style="cyan", width=15)
        table.add_column("Status", style="green", width=15)
        table.add_column("Description", style="yellow")
        
        # Stages in order
        for stage_name in ["extraction", "inventory", "pricing", "fulfillment"]:
            events = self.event_history.get(stage_name, [])
            
            if events:
                latest = events[-1]
                emoji = self.STAGE_EMOJI.get(latest.stage, "•")
                
                table.add_row(
                    stage_name.capitalize(),
                    f"{emoji} {latest.stage}",
                    latest.description[:50]  # Truncate if too long
                )
            else:
                table.add_row(
                    stage_name.capitalize(),
                    "⏱️  waiting",
                    "Not started"
                )
        
        self.console.print(table)
    
    def _render_recent_events(self) -> None:
        """Display recent workflow events."""
        self.console.print()
        self.console.print("[bold]📋 Recent Events[/bold]:")
        
        # Collect all events sorted by timestamp
        all_events = []
        for events_list in self.event_history.values():
            all_events.extend(events_list)
        
        all_events.sort(key=lambda e: e.timestamp)
        recent = all_events[-self.max_history:]
        
        if not recent:
            self.console.print("  [dim]No events yet[/dim]")
            return
        
        for event in recent:
            emoji = self.STAGE_EMOJI.get(event.stage, "•")
            time_str = event.timestamp.strftime("%H:%M:%S")
            color = self.EVENT_COLORS.get(event.event_type, "white")
            
            details_str = ""
            if event.item_id:
                details_str = f" ({event.item_id})"
            
            self.console.print(
                f"  {emoji} [{time_str}] "
                f"[{color}]{event.event_type.upper()}[/{color}] "
                f"{event.description}{details_str}"
            )
    
    def _render_request_summary(self) -> None:
        """Display summary of requests processed."""
        self.console.print()
        self.console.print("[bold]📦 Requests Processed[/bold]:")
        
        if not self.latest_by_request:
            self.console.print("  [dim]No requests processed yet[/dim]")
            return
        
        # Group by request
        requests_summary = defaultdict(lambda: {
            "extraction": "⏱️ ",
            "inventory": "⏱️ ",
            "pricing": "⏱️ ",
            "fulfillment": "⏱️ "
        })
        
        for events in self.event_history.values():
            for event in events:
                requests_summary[event.request_id][event.event_type] = (
                    self.STAGE_EMOJI.get(event.stage, "•")
                )
        
        # Display top 5 requests
        for request_id, stages in list(requests_summary.items())[:5]:
            status = " ".join([
                stages["extraction"],
                stages["inventory"],
                stages["pricing"],
                stages["fulfillment"]
            ])
            self.console.print(f"  {request_id}: {status}")
    
    def _update_request_status(self, request_id: str, event: WorkflowEvent) -> None:
        """Update the status of a request."""
        if request_id not in self.request_status:
            self.request_status[request_id] = "processing"
        
        if event.stage == "failed":
            self.request_status[request_id] = "failed"
        elif event.event_type == "fulfillment" and event.stage == "completed":
            self.request_status[request_id] = "completed"
    
    def get_summary(self) -> dict:
        """
        Get a summary of all workflow activity.
        Useful for final reporting.
        
        Returns:
            Dictionary with workflow statistics
        """
        all_events = []
        for events_list in self.event_history.values():
            all_events.extend(events_list)
        
        completed_count = sum(1 for e in all_events if e.stage == "completed")
        failed_count = sum(1 for e in all_events if e.stage == "failed")
        
        return {
            "total_events": len(all_events),
            "completed_stages": completed_count,
            "failed_stages": failed_count,
            "requests_touched": len(self.latest_by_request),
            "event_types": list(self.event_history.keys())
        }
