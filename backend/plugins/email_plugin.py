#!/usr/bin/env python3
"""
================================================================================
VEGA EMAIL PLUGIN - Email Integration
================================================================================
Email operations via IMAP/SMTP:
- Send emails with attachments
- Read/fetch emails
- Search emails
- Folder management
- Multi-provider support (Gmail, Outlook, custom)

References:
- OpenJarvis/OpenJarvis: Email plugin pattern
- devansh2506/JARVIS-A-Multi-Agentic-Personal-Assistant: Email agent
================================================================================
"""

from __future__ import annotations

import asyncio
import imaplib
import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Any

from plugins.plugin_loader import PluginBase, PluginConfig, PluginMetadata
from tools.tool_registry import ToolMetadata, ToolParameter
from tools.builtin.file_tools import _workspace
from config import get_settings, settings

logger = logging.getLogger("vega.plugins.email")


# =============================================================================
# PLUGIN METADATA
# =============================================================================

PLUGIN_METADATA = {
    "name": "email",
    "version": "1.0.0",
    "description": "Email integration (IMAP/SMTP) for Gmail, Outlook, and custom servers",
    "author": "VEGA Team",
    "license": "MIT",
    "requires": [],
    "dependencies": [],
    "entry_point": "email_plugin.py",
    "class_name": "EmailPlugin",
    "config_schema": {
        "provider": {"type": "string", "description": "Email provider: gmail, outlook, custom", "default": "gmail"},
        "smtp_host": {"type": "string", "description": "SMTP host"},
        "smtp_port": {"type": "integer", "description": "SMTP port", "default": 587},
        "imap_host": {"type": "string", "description": "IMAP host"},
        "imap_port": {"type": "integer", "description": "IMAP port", "default": 993},
        "username": {"type": "string", "description": "Email username"},
        "password": {"type": "string", "description": "Email password/app password"},
        "use_ssl": {"type": "boolean", "description": "Use SSL for IMAP", "default": True},
    },
    "default_config": {
        "enabled": False,
    },
    "permissions": ["email.send", "email.read", "email.search"],
    "sandboxed": True,
}


# =============================================================================
# EMAIL PLUGIN
# =============================================================================

class EmailPlugin(PluginBase):
    """Email integration plugin."""

    def __init__(self, config: PluginConfig, event_bus: Any = None, tool_registry: Any = None):
        super().__init__(config, event_bus, tool_registry)
        self._provider = config.config.get("provider", "gmail")
        self._smtp_host = config.config.get("smtp_host")
        self._smtp_port = config.config.get("smtp_port", 587)
        self._imap_host = config.config.get("imap_host")
        self._imap_port = config.config.get("imap_port", 993)
        self._username = config.config.get("username")
        self._password = config.config.get("password")
        self._use_ssl = config.config.get("use_ssl", True)

        # Provider defaults
        self._load_provider_defaults()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(**PLUGIN_METADATA)

    def _load_provider_defaults(self) -> None:
        """Load provider-specific defaults."""
        providers = {
            "gmail": {
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "imap_host": "imap.gmail.com",
                "imap_port": 993,
            },
            "outlook": {
                "smtp_host": "smtp.office365.com",
                "smtp_port": 587,
                "imap_host": "outlook.office365.com",
                "imap_port": 993,
            },
        }

        defaults = providers.get(self._provider, {})
        for key, value in defaults.items():
            if not getattr(self, f"_{key}", None):
                setattr(self, f"_{key}", value)

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(**PLUGIN_METADATA)

    async def initialize(self) -> bool:
        if not self._username or not self._password:
            logger.warning("Email credentials not configured")
            return True

        logger.info("EmailPlugin initialized")
        return True

    async def shutdown(self) -> None:
        await super().shutdown()

    async def get_tools(self) -> list:
        return [
            ("email_send", self._tool_send, ToolMetadata(
                name="email_send",
                description="Send email with optional attachments",
                category="communication",
                parameters=[
                    ToolParameter("to", "array", "Recipient email(s)", required=True),
                    ToolParameter("subject", "string", "Email subject", required=True),
                    ToolParameter("body", "string", "Email body", required=True),
                    ToolParameter("cc", "array", "CC recipients"),
                    ToolParameter("bcc", "array", "BCC recipients"),
                    ToolParameter("attachments", "array", "Attachment file paths"),
                    ToolParameter("html", "boolean", "Body is HTML", default=False),
                ],
                requires_approval=True,
            )),
            ("email_read", self._tool_read, ToolMetadata(
                name="email_read",
                description="Read/fetch emails from IMAP",
                category="communication",
                parameters=[
                    ToolParameter("folder", "string", "Mail folder", default="INBOX"),
                    ToolParameter("limit", "integer", "Max emails", default=20),
                    ToolParameter("unread_only", "boolean", "Only unread", default=False),
                    ToolParameter("search", "string", "Search query"),
                ],
                requires_approval=False,
            )),
            ("email_search", self._tool_search, ToolMetadata(
                name="email_search",
                description="Search emails",
                category="communication",
                parameters=[
                    ToolParameter("query", "string", "Search query", required=True),
                    ToolParameter("folder", "string", "Mail folder", default="INBOX"),
                    ToolParameter("limit", "integer", "Max results", default=20),
                ],
                requires_approval=False,
            )),
        ]

    # -------------------------------------------------------------------------
    # SMTP
    # -------------------------------------------------------------------------

    def _create_smtp_connection(self):
        """Create SMTP connection."""
        context = ssl.create_default_context()
        server = smtplib.SMTP(self._smtp_host, self._smtp_port)
        server.starttls(context=context)
        server.login(self._username, self._password)
        return server

    # -------------------------------------------------------------------------
    # IMAP
    # -------------------------------------------------------------------------

    def _create_imap_connection(self):
        """Create IMAP connection."""
        mail = imaplib.IMAP4_SSL(self._imap_host, self._imap_port)
        mail.login(self._username, self._password)
        return mail

    # -------------------------------------------------------------------------
    # TOOL IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    async def _tool_send(self, params: dict) -> dict:
        to = params.get("to", [])
        subject = params.get("subject", "")
        body = params.get("body", "")
        cc = params.get("cc", [])
        bcc = params.get("bcc", [])
        attachments = params.get("attachments", [])
        html = params.get("html", False)

        if not to or not subject:
            return {"success": False, "error": "Recipient and subject required"}

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self._username
            msg["To"] = ", ".join(to) if isinstance(to, list) else to
            if cc:
                msg["Cc"] = ", ".join(cc) if isinstance(cc, list) else cc
            msg["Subject"] = subject

            # Body
            if html:
                msg.attach(MIMEText(body, "html"))
            else:
                msg.attach(MIMEText(body, "plain"))

            # Attachments
            for att_path in attachments:
                file_path = _workspace.resolve_path(att_path)
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f"attachment; filename={file_path.name}")
                    msg.attach(part)

            # Send
            server = self._create_smtp_connection()
            all_recipients = to + cc + bcc
            if isinstance(all_recipients, str):
                all_recipients = [all_recipients]

            server.sendmail(self._username, all_recipients, msg.as_string())
            server.quit()

            return {"success": True, "message": "Email sent", "recipients": len(all_recipients)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_read(self, params: dict) -> dict:
        folder = params.get("folder", "INBOX")
        limit = params.get("limit", 20)
        unread_only = params.get("unread_only", False)
        search = params.get("search", "")

        try:
            mail = self._create_imap_connection()
            mail.select(folder)

            # Build search criteria
            criteria = ["ALL"]
            if unread_only:
                criteria = ["UNSEEN"]
            if search:
                criteria = ["SEARCH", search]

            status, messages = mail.search(None, *criteria)
            if status != "OK":
                return {"success": False, "error": "Search failed"}

            email_ids = messages[0].split()[-limit:]
            emails = []

            for eid in reversed(email_ids):
                status, msg_data = mail.fetch(eid, "(RFC822)")
                if status != "OK":
                    continue

                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                emails.append({
                    "id": eid.decode(),
                    "subject": msg.get("Subject", ""),
                    "from": msg.get("From", ""),
                    "to": msg.get("To", ""),
                    "date": msg.get("Date", ""),
                    "body": self._extract_email_body(msg),
                })

            mail.close()
            mail.logout()

            return {"success": True, "emails": emails, "count": len(emails)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_email_body(self, msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode(errors="ignore")
        else:
            return msg.get_payload(decode=True).decode(errors="ignore")
        return ""

    async def _tool_search(self, params: dict) -> dict:
        query = params.get("query", "")
        folder = params.get("folder", "INBOX")
        limit = params.get("limit", 20)

        params_read = {
            "folder": folder,
            "limit": limit,
            "search": query,
        }
        return await self._tool_read(params_read)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EmailPlugin",
    "PLUGIN_METADATA",
]