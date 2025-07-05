"""
Sovereign AI Agent - Graphical User Interface

A modern, beautiful desktop application built with CustomTkinter that provides
a seamless interface for interacting with the Sovereign AI Agent's dual-model
architecture, voice interface, and advanced features.

Features:
- Modern chat interface with message bubbles
- Real-time status indicators (listening, thinking, speaking)
- Model switching indicators (Talker/Thinker)
- Settings panel with theme customization
- Voice interface controls
- Screen capture toggle
- System tray integration
- Keyboard shortcuts
- Responsive design
"""

import asyncio
import threading
import logging
import queue
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import sys
import os
import json

# GUI imports
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import pystray
from PIL import Image, ImageTk

# Backend imports
from .config import Config, config
from .logger import setup_logger, get_performance_logger
from .orchestrator import ModelOrchestrator, QueryContext, ModelChoice, QueryComplexity
from .hardware import hardware_detector

# Optional voice interface
try:
    from .voice_interface import VoiceInterfaceManager, VoiceState
    VOICE_AVAILABLE = True
except ImportError:
    VoiceInterfaceManager = None
    VoiceState = None
    VOICE_AVAILABLE = False

# Configure CustomTkinter
ctk.set_appearance_mode("dark")  # Default to dark theme
ctk.set_default_color_theme("blue")  # Default color theme


class SovereignGUI:
    """Main GUI application for Sovereign AI Agent"""
    
    def __init__(self):
        """Initialize the GUI application"""
        self.config = config
        self.logger = logging.getLogger("sovereign.gui")
        
        # Core components
        self.orchestrator: Optional[ModelOrchestrator] = None
        self.voice_manager: Optional[VoiceInterfaceManager] = None
        
        # GUI state
        self.is_listening = False
        self.is_thinking = False
        self.is_speaking = False
        self.current_model = ModelChoice.TALKER
        self.theme = "dark"
        
        # Message history
        self.message_history: List[Dict[str, Any]] = []
        
        # GUI components
        self.root: Optional[ctk.CTk] = None
        self.chat_frame: Optional[ctk.CTkScrollableFrame] = None
        self.input_frame: Optional[ctk.CTkFrame] = None
        self.status_frame: Optional[ctk.CTkFrame] = None
        self.settings_frame: Optional[ctk.CTkFrame] = None
        
        # Input components
        self.message_entry: Optional[ctk.CTkTextbox] = None
        self.send_button: Optional[ctk.CTkButton] = None
        self.voice_button: Optional[ctk.CTkButton] = None
        
        # Status indicators
        self.status_label: Optional[ctk.CTkLabel] = None
        self.model_label: Optional[ctk.CTkLabel] = None
        self.connection_label: Optional[ctk.CTkLabel] = None
        
        # Settings components
        self.theme_switch: Optional[ctk.CTkSwitch] = None
        self.voice_switch: Optional[ctk.CTkSwitch] = None
        self.screen_capture_switch: Optional[ctk.CTkSwitch] = None
        
        # System tray
        self.tray_icon: Optional[pystray.Icon] = None
        self.tray_thread: Optional[threading.Thread] = None
        
        # Event loop for async operations
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        
        # Thread-safe communication
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.is_processing = False
        
    async def initialize_backend(self):
        """Initialize the backend components"""
        try:
            self.logger.info("🔧 Initializing backend components...")
            
            # Initialize orchestrator
            self.orchestrator = ModelOrchestrator(self.config)
            await self.orchestrator.initialize()
            self.logger.info("✅ Orchestrator initialized")
            
            # Initialize voice interface if available
            if VOICE_AVAILABLE and self.config.voice.enabled:
                try:
                    self.voice_manager = VoiceInterfaceManager(self.config)
                    success = await self.voice_manager.initialize()
                    if success:
                        self.logger.info("✅ Voice interface initialized")
                        
                        # Add voice callbacks
                        self.voice_manager.add_voice_callback('on_wake_word', self._on_wake_word)
                        self.voice_manager.add_voice_callback('on_speech_recognized', self._on_speech_recognized)
                        self.voice_manager.add_voice_callback('on_state_change', self._on_voice_state_change)
                    else:
                        self.logger.warning("⚠️ Voice interface initialization failed")
                        self.voice_manager = None
                except Exception as e:
                    self.logger.error(f"❌ Voice interface error: {e}")
                    self.voice_manager = None
            
            self.logger.info("🚀 Backend initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Backend initialization failed: {e}")
            return False
    
    def process_request_thread(self, prompt: str):
        """Worker function to process AI requests in a separate thread"""
        try:
            self.logger.info(f"🧠 Processing request in worker thread: {prompt[:50]}...")
            
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Process the query using the orchestrator
                context = QueryContext(
                    query=prompt,
                    conversation_history=self.message_history[-10:],  # Last 10 messages for context
                    model_choice=self.current_model,
                    include_screen_context=self.config.screen_capture.enabled,
                    voice_input=False
                )
                
                # Run the async query processing
                result = loop.run_until_complete(self.orchestrator.process_query(context))
                
                # Put the result in the response queue
                self.response_queue.put({
                    'success': True,
                    'response': result.response,
                    'model_used': result.model_used,
                    'complexity': result.complexity,
                    'processing_time': result.processing_time
                })
                
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"❌ Worker thread error: {e}")
            # Put error in response queue
            self.response_queue.put({
                'success': False,
                'error': str(e)
            })
        finally:
            # Always mark as no longer processing
            self.is_processing = False
    
    def check_for_responses(self):
        """Check for responses from worker threads and update UI"""
        try:
            # Non-blocking check for responses
            response = self.response_queue.get_nowait()
            
            if response['success']:
                # Add AI response to chat
                model_name = response['model_used'].value if response['model_used'] else "unknown"
                self._add_message_bubble(
                    response['response'],
                    sender="ai",
                    model=model_name
                )
                
                # Add to message history
                self.message_history.append({
                    'message': response['response'],
                    'sender': 'ai',
                    'timestamp': datetime.now(),
                    'model': model_name
                })
                
                # Update status
                self._update_status("🤖 Ready")
                self._update_model_indicator(response['model_used'] if response['model_used'] else ModelChoice.TALKER)
                
                # Re-enable send button
                if self.send_button:
                    self.send_button.configure(state="normal", text="Send")
                    
            else:
                # Handle error
                self._add_system_message(f"Error: {response['error']}")
                self._update_status("❌ Error")
                
                # Re-enable send button
                if self.send_button:
                    self.send_button.configure(state="normal", text="Send")
                    
        except queue.Empty:
            # No response yet, continue checking
            pass
        except Exception as e:
            self.logger.error(f"❌ Error checking responses: {e}")
            
        # Schedule next check
        if self.root:
            self.root.after(100, self.check_for_responses)
    
    def setup_gui(self):
        """Setup the main GUI window and components"""
        self.logger.info("🎨 Setting up GUI...")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Sovereign AI Agent")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set window icon (if available)
        try:
            icon_path = Path(__file__).parent.parent.parent / "assets" / "icon.png"
            if icon_path.exists():
                icon = ImageTk.PhotoImage(Image.open(icon_path))
                self.root.iconphoto(True, icon)
        except Exception:
            pass  # No icon available
        
        # Configure grid weights for responsiveness
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create main container
        main_container = ctk.CTkFrame(self.root)
        main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_rowconfigure(1, weight=1)
        
        # Create components
        self._create_status_bar(main_container)
        self._create_chat_interface(main_container)
        self._create_input_interface(main_container)
        self._create_settings_panel(main_container)
        
        # Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()
        
        # Setup window protocols
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        # Start response checking
        self.check_for_responses()
        
        self.logger.info("✅ GUI setup complete")
    
    def _create_status_bar(self, parent):
        """Create the status bar with indicators"""
        self.status_frame = ctk.CTkFrame(parent)
        self.status_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(0, 5))
        self.status_frame.grid_columnconfigure(1, weight=1)
        
        # AI Status indicator
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="🤖 Ready",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Model indicator
        self.model_label = ctk.CTkLabel(
            self.status_frame,
            text="💬 Talker",
            font=ctk.CTkFont(size=12)
        )
        self.model_label.grid(row=0, column=1, padx=10, pady=5)
        
        # Connection status
        self.connection_label = ctk.CTkLabel(
            self.status_frame,
            text="🔗 Connected",
            font=ctk.CTkFont(size=12)
        )
        self.connection_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")
        
        # Settings toggle button
        settings_btn = ctk.CTkButton(
            self.status_frame,
            text="⚙️",
            width=30,
            command=self._toggle_settings_panel
        )
        settings_btn.grid(row=0, column=3, padx=5, pady=5)
    
    def _create_chat_interface(self, parent):
        """Create the main chat interface"""
        chat_container = ctk.CTkFrame(parent)
        chat_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        chat_container.grid_columnconfigure(0, weight=1)
        chat_container.grid_rowconfigure(0, weight=1)
        
        # Scrollable chat area
        self.chat_frame = ctk.CTkScrollableFrame(
            chat_container,
            corner_radius=10
        )
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_frame.grid_columnconfigure(0, weight=1)
        
        # Welcome message
        self._add_system_message("Welcome to Sovereign AI Agent! 🚀\n\nType a message or click the microphone to start speaking.")
    
    def _create_input_interface(self, parent):
        """Create the input interface with text entry and voice button"""
        self.input_frame = ctk.CTkFrame(parent)
        self.input_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.input_frame.grid_columnconfigure(1, weight=1)
        
        # Voice button
        voice_text = "🎤" if VOICE_AVAILABLE else "🎤❌"
        self.voice_button = ctk.CTkButton(
            self.input_frame,
            text=voice_text,
            width=50,
            command=self._toggle_voice_listening,
            state="normal" if VOICE_AVAILABLE else "disabled"
        )
        self.voice_button.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="w")
        
        # Message input (multi-line)
        self.message_entry = ctk.CTkTextbox(
            self.input_frame,
            height=80,
            wrap="word",
            font=ctk.CTkFont(size=14)
        )
        self.message_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        
        # Send button
        self.send_button = ctk.CTkButton(
            self.input_frame,
            text="Send",
            width=80,
            command=self._send_message
        )
        self.send_button.grid(row=0, column=2, padx=(5, 10), pady=10, sticky="e")
        
        # Bind Enter key to send (Shift+Enter for new line)
        self.message_entry.bind("<Return>", self._on_enter_key)
        self.message_entry.bind("<Shift-Return>", lambda e: None)  # Allow Shift+Enter for newlines
    
    def _create_settings_panel(self, parent):
        """Create the settings panel (initially hidden)"""
        self.settings_frame = ctk.CTkFrame(parent)
        # Initially not gridded (hidden)
        self.settings_frame.grid_columnconfigure(0, weight=1)
        
        # Settings title
        title_label = ctk.CTkLabel(
            self.settings_frame,
            text="Settings",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Theme settings
        theme_frame = ctk.CTkFrame(self.settings_frame)
        theme_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        theme_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(theme_frame, text="Dark Theme:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.theme_switch = ctk.CTkSwitch(
            theme_frame,
            text="",
            command=self._toggle_theme
        )
        self.theme_switch.grid(row=0, column=1, padx=10, pady=10, sticky="e")
        self.theme_switch.select() if self.theme == "dark" else self.theme_switch.deselect()
        
        # Voice settings
        if VOICE_AVAILABLE:
            voice_frame = ctk.CTkFrame(self.settings_frame)
            voice_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
            voice_frame.grid_columnconfigure(1, weight=1)
            
            ctk.CTkLabel(voice_frame, text="Voice Interface:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
            self.voice_switch = ctk.CTkSwitch(
                voice_frame,
                text="",
                command=self._toggle_voice_interface
            )
            self.voice_switch.grid(row=0, column=1, padx=10, pady=10, sticky="e")
            self.voice_switch.select() if self.config.voice.enabled else self.voice_switch.deselect()
        
        # Screen capture settings
        screen_frame = ctk.CTkFrame(self.settings_frame)
        screen_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=10)
        screen_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(screen_frame, text="Screen Capture:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.screen_capture_switch = ctk.CTkSwitch(
            screen_frame,
            text="",
            command=self._toggle_screen_capture
        )
        self.screen_capture_switch.grid(row=0, column=1, padx=10, pady=10, sticky="e")
        self.screen_capture_switch.select() if self.config.screen_capture.enabled else self.screen_capture_switch.deselect()
        
        # Action buttons
        button_frame = ctk.CTkFrame(self.settings_frame)
        button_frame.grid(row=4, column=0, sticky="ew", padx=20, pady=20)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        reset_btn = ctk.CTkButton(
            button_frame,
            text="Reset to Defaults",
            command=self._reset_settings
        )
        reset_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        close_btn = ctk.CTkButton(
            button_frame,
            text="Close Settings",
            command=self._toggle_settings_panel
        )
        close_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        if self.root:
            # Ctrl+N: New conversation
            self.root.bind("<Control-n>", lambda e: self._new_conversation())
            
            # Ctrl+S: Save conversation
            self.root.bind("<Control-s>", lambda e: self._save_conversation())
            
            # Ctrl+O: Open conversation
            self.root.bind("<Control-o>", lambda e: self._load_conversation())
            
            # Ctrl+Comma: Settings
            self.root.bind("<Control-comma>", lambda e: self._toggle_settings_panel())
            
            # F1: Help
            self.root.bind("<F1>", lambda e: self._show_help())
            
            # ESC: Stop current operation
            self.root.bind("<Escape>", lambda e: self._stop_current_operation())
    
    def _add_message_bubble(self, message: str, sender: str = "user", timestamp: Optional[datetime] = None, model: Optional[str] = None):
        """Add a message bubble to the chat interface"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create message frame
        msg_frame = ctk.CTkFrame(self.chat_frame)
        msg_frame.grid(sticky="ew", padx=10, pady=5)
        msg_frame.grid_columnconfigure(0, weight=1)
        
        # Configure grid weight for the chat frame
        current_row = len(self.chat_frame.winfo_children())
        self.chat_frame.grid_rowconfigure(current_row, weight=0)
        
        # Header with sender and timestamp
        header_frame = ctk.CTkFrame(msg_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Sender icon and name
        if sender == "user":
            icon = "👤"
            sender_text = "You"
            text_color = ("gray10", "white")
        else:
            icon = "🤖"
            sender_text = f"Sovereign ({model or 'AI'})"
            text_color = ("blue", "lightblue")
        
        sender_label = ctk.CTkLabel(
            header_frame,
            text=f"{icon} {sender_text}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=text_color
        )
        sender_label.grid(row=0, column=0, sticky="w")
        
        # Timestamp
        time_label = ctk.CTkLabel(
            header_frame,
            text=timestamp.strftime("%H:%M:%S"),
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray50")
        )
        time_label.grid(row=0, column=1, sticky="e")
        
        # Message content
        content_frame = ctk.CTkFrame(msg_frame)
        content_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        content_frame.grid_columnconfigure(0, weight=1)
        
        # Format message (handle code blocks, lists, etc.)
        formatted_message = self._format_message_content(message)
        
        message_label = ctk.CTkTextbox(
            content_frame,
            height=max(50, min(200, len(message.split('\n')) * 20)),
            wrap="word",
            font=ctk.CTkFont(size=14)
        )
        message_label.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        message_label.insert("1.0", formatted_message)
        message_label.configure(state="disabled")  # Make read-only
        
        # Auto-scroll to bottom
        self.root.after(100, lambda: self.chat_frame._parent_canvas.yview_moveto(1.0))
        
        # Store message in history
        self.message_history.append({
            "message": message,
            "sender": sender,
            "timestamp": timestamp,
            "model": model
        })
    
    def _add_system_message(self, message: str):
        """Add a system message to the chat"""
        msg_frame = ctk.CTkFrame(self.chat_frame, fg_color=("orange", "darkorange"))
        msg_frame.grid(sticky="ew", padx=10, pady=5)
        msg_frame.grid_columnconfigure(0, weight=1)
        
        system_label = ctk.CTkLabel(
            msg_frame,
            text=f"ℹ️ {message}",
            font=ctk.CTkFont(size=12),
            wraplength=800,
            justify="left"
        )
        system_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")
        
        # Auto-scroll to bottom
        self.root.after(100, lambda: self.chat_frame._parent_canvas.yview_moveto(1.0))
    
    def _format_message_content(self, message: str) -> str:
        """Format message content (code blocks, lists, etc.)"""
        # For now, return as-is. Could be enhanced with markdown formatting
        return message
    
    def _update_status(self, status: str, color: Optional[str] = None):
        """Update the status indicator"""
        if self.status_label:
            self.status_label.configure(text=status)
            if color:
                self.status_label.configure(text_color=color)
    
    def _update_model_indicator(self, model: ModelChoice):
        """Update the model indicator"""
        if self.model_label:
            if model == ModelChoice.TALKER:
                self.model_label.configure(text="💬 Talker", text_color=("blue", "lightblue"))
            elif model == ModelChoice.THINKER:
                self.model_label.configure(text="🧠 Thinker", text_color=("purple", "mediumpurple"))
            else:
                self.model_label.configure(text="🔄 Processing", text_color=("orange", "darkorange"))
    
    def _send_message(self):
        """Send message using worker thread"""
        message = self.message_entry.get("1.0", "end").strip()
        if not message:
            return
        
        # Check if already processing
        if self.is_processing:
            self.logger.warning("Already processing a request, please wait...")
            return
            
        # Check if orchestrator is available
        if not self.orchestrator:
            self._add_system_message("❌ AI backend not initialized. Please restart the application.")
            return
        
        # Clear input
        self.message_entry.delete("1.0", "end")
        
        # Add user message to chat
        self._add_message_bubble(message, "user")
        
        # Add to message history
        self.message_history.append({
            'message': message,
            'sender': 'user',
            'timestamp': datetime.now(),
            'model': None
        })
        
        # Update status and disable send button
        self._update_status("🧠 Thinking...")
        if self.send_button:
            self.send_button.configure(state="disabled", text="Processing...")
        
        # Mark as processing
        self.is_processing = True
        
        # Start worker thread
        worker = threading.Thread(
            target=self.process_request_thread,
            args=(message,),
            daemon=True
        )
        worker.start()
    
    def _on_enter_key(self, event):
        """Handle Enter key in message entry"""
        if event.state & 0x1:  # Shift is held
            return "break"  # Allow Shift+Enter for newlines
        else:
            self._send_message()
            return "break"  # Prevent default behavior
    
    def _toggle_voice_listening(self):
        """Toggle voice listening mode"""
        if not VOICE_AVAILABLE or not self.voice_manager:
            messagebox.showwarning("Voice Not Available", "Voice interface is not available or not properly initialized.")
            return
        
        if self.is_listening:
            # Stop listening
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.voice_manager.stop_listening(),
                    self.loop
                )
            self.is_listening = False
            self.voice_button.configure(text="🎤")
            self._update_status("🤖 Ready")
        else:
            # Start listening
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.voice_manager.start_listening(),
                    self.loop
                )
            self.is_listening = True
            self.voice_button.configure(text="🔴")
            self._update_status("👂 Listening...", ("red", "lightcoral"))
    
    def _toggle_settings_panel(self):
        """Toggle the settings panel visibility"""
        try:
            if self.settings_frame.winfo_manager():  # If it's currently shown
                self.settings_frame.grid_remove()
            else:
                self.settings_frame.grid(row=1, column=1, sticky="ns", padx=(0, 10), pady=5)
                # Adjust main chat area
                self.root.grid_columnconfigure(1, weight=0, minsize=300)
        except:
            pass
    
    def _toggle_theme(self):
        """Toggle between light and dark themes"""
        if self.theme == "dark":
            self.theme = "light"
            ctk.set_appearance_mode("light")
        else:
            self.theme = "dark"
            ctk.set_appearance_mode("dark")
    
    def _toggle_voice_interface(self):
        """Toggle voice interface on/off"""
        self.config.voice.enabled = not self.config.voice.enabled
        # Could reinitialize voice interface here if needed
    
    def _toggle_screen_capture(self):
        """Toggle screen capture on/off"""
        self.config.screen_capture.enabled = not self.config.screen_capture.enabled
    
    def _reset_settings(self):
        """Reset all settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to defaults?"):
            # Reset theme
            self.theme = "dark"
            ctk.set_appearance_mode("dark")
            self.theme_switch.select()
            
            # Reset other settings
            self.config.voice.enabled = True
            self.config.screen_capture.enabled = False
            
            # Update switches
            if self.voice_switch:
                self.voice_switch.select()
            self.screen_capture_switch.deselect()
    
    def _new_conversation(self):
        """Start a new conversation"""
        if messagebox.askyesno("New Conversation", "Are you sure you want to start a new conversation? Current history will be cleared."):
            # Clear chat
            for widget in self.chat_frame.winfo_children():
                widget.destroy()
            
            # Clear history
            self.message_history.clear()
            
            # Add welcome message
            self._add_system_message("New conversation started! 🚀")
    
    def _save_conversation(self):
        """Save current conversation to file"""
        if not self.message_history:
            messagebox.showinfo("No Conversation", "No conversation to save.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Conversation"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.message_history, f, indent=2, default=str)
                messagebox.showinfo("Saved", f"Conversation saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save conversation: {e}")
    
    def _load_conversation(self):
        """Load conversation from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Conversation"
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                # Clear current chat
                for widget in self.chat_frame.winfo_children():
                    widget.destroy()
                
                # Load messages
                self.message_history = history
                for msg in history:
                    timestamp = datetime.fromisoformat(msg['timestamp']) if isinstance(msg['timestamp'], str) else msg['timestamp']
                    self._add_message_bubble(
                        msg['message'],
                        msg['sender'],
                        timestamp,
                        msg.get('model')
                    )
                
                messagebox.showinfo("Loaded", f"Conversation loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load conversation: {e}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
Sovereign AI Agent - Help

Keyboard Shortcuts:
• Ctrl+N: New conversation
• Ctrl+S: Save conversation
• Ctrl+O: Load conversation
• Ctrl+,: Settings
• F1: This help
• ESC: Stop current operation
• Enter: Send message
• Shift+Enter: New line in message

Voice Commands:
• Click microphone button to start/stop listening
• Say "Hey Sovereign" for wake word activation (if enabled)

Features:
• Dual AI models (Talker for quick responses, Thinker for complex tasks)
• Voice interface with speech recognition and text-to-speech
• Screen capture integration
• Theme customization
• Conversation history and export

For more information, visit the project documentation.
        """
        
        messagebox.showinfo("Help", help_text)
    
    def _stop_current_operation(self):
        """Stop current AI operation"""
        # Could implement operation cancellation here
        self._update_status("🤖 Ready")
        if self.is_listening:
            self._toggle_voice_listening()
    
    # Voice interface callbacks
    def _on_wake_word(self, data):
        """Handle wake word detection"""
        self.root.after(0, lambda: self._update_status("👂 Wake word detected!", ("green", "lightgreen")))
    
    def _on_speech_recognized(self, data):
        """Handle speech recognition result"""
        if 'text' in data and data['text'].strip():
            # Add recognized text to input
            self.root.after(0, lambda: self.message_entry.insert("1.0", data['text']))
            # Auto-send if configured
            if self.config.voice.auto_send_speech:
                self.root.after(100, self._send_message)
    
    def _on_voice_state_change(self, data):
        """Handle voice state changes"""
        state = data.get('new_state', '')
        if state == 'listening':
            self.root.after(0, lambda: self._update_status("👂 Listening...", ("blue", "lightblue")))
        elif state == 'processing':
            self.root.after(0, lambda: self._update_status("🔄 Processing speech...", ("orange", "darkorange")))
        elif state == 'speaking':
            self.root.after(0, lambda: self._update_status("🗣️ Speaking...", ("purple", "mediumpurple")))
        else:
            self.root.after(0, lambda: self._update_status("🤖 Ready"))
    
    def _on_window_close(self):
        """Handle window close event"""
        if messagebox.askyesno("Quit", "Are you sure you want to quit Sovereign AI Agent?"):
            self._cleanup_and_exit()
    
    def _cleanup_and_exit(self):
        """Clean up resources and exit"""
        try:
            # Stop voice interface
            if self.voice_manager and self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.voice_manager.stop_listening(),
                    self.loop
                ).result(timeout=2)
            
            # Close orchestrator
            if self.orchestrator and self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.orchestrator.close(),
                    self.loop
                ).result(timeout=2)
            
            # Stop event loop
            if self.loop and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.loop.stop)
            
            # Stop system tray
            if self.tray_icon:
                self.tray_icon.stop()
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        
        finally:
            if self.root:
                self.root.quit()
                self.root.destroy()
            sys.exit(0)
    
    def _run_event_loop(self):
        """Run the asyncio event loop in a separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            # Initialize backend
            init_success = self.loop.run_until_complete(self.initialize_backend())
            if not init_success:
                self.logger.error("Failed to initialize backend")
                return
            
            # Run the event loop
            self.loop.run_forever()
        except Exception as e:
            self.logger.error(f"Event loop error: {e}")
        finally:
            self.loop.close()
    
    def run(self):
        """Run the GUI application"""
        try:
            # Setup GUI
            self.setup_gui()
            
            # Start event loop thread
            self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self.loop_thread.start()
            
            # Wait for backend initialization
            import time
            time.sleep(2)  # Give backend time to initialize
            
            # Update connection status
            self._update_status("🤖 Ready", ("green", "lightgreen"))
            
            # Start the GUI main loop
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(f"GUI error: {e}")
            messagebox.showerror("Error", f"GUI error: {e}")
        finally:
            self._cleanup_and_exit()


def run_gui():
    """Entry point for running the GUI application"""
    try:
        # Setup logging
        logger = setup_logger("sovereign.gui", log_level="INFO")
        logger.info("🚀 Starting Sovereign AI Agent GUI...")
        
        # Create and run GUI
        app = SovereignGUI()
        app.run()
        
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_gui() 