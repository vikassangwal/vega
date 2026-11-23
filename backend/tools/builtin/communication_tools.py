"""
JARVIS OMEGA - Communication Tools
Built-in tools for email, messaging, notifications, calendar, contacts.
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
from pathlib import Path
from typing import Any

from tools.tool_registry import ToolRegistry, ToolMetadata, ToolParameter
from tools.builtin.file_tools import _workspace
from config import get_settings, settings

logger = logging.getLogger("vega.tools.communication")


# =============================================================================
# EMAIL TOOLS
# =============================================================================

async def email_send(params: dict) -> dict:
    to = params.get("to", [])
    subject = params.get("subject", "")
    body = params.get("body", "")
    cc = params.get("cc", [])
    bcc = params.get("bcc", [])
    attachments = params.get("attachments", [])
    provider = params.get("provider", "smtp")

    if not to or not subject:
        return {"success": False, "error": "Recipient and subject required"}

    smtp_config = _get_smtp_config(provider)
    if not smtp_config:
        return {"success": False, "error": f"SMTP config not found for {provider}"}

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_config["username"]
        msg["To"] = ", ".join(to) if isinstance(to, list) else to
        if cc:
            msg["Cc"] = ", ".join(cc) if isinstance(cc, list) else cc
        msg["Subject"] = subject

        if params.get("html", False):
            msg.attach(MIMEText(body, "html"))
        else:
            msg.attach(MIMEText(body, "plain"))

        for att_path in attachments:
            file_path = _workspace.resolve_path(att_path)
            if file_path.exists():
                with open(file_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={file_path.name}")
                msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_config["host"], smtp_config["port"]) as server:
            server.starttls(context=context)
            server.login(smtp_config["username"], smtp_config["password"])

            all_recipients = to + cc + bcc
            if isinstance(all_recipients, str):
                all_recipients = [all_recipients]

            server.sendmail(smtp_config["username"], all_recipients, msg.as_string())

        return {"success": True, "message": "Email sent", "recipients": len(all_recipients)}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def email_read(params: dict) -> dict:
    folder = params.get("folder", "INBOX")
    limit = params.get("limit", 20)
    unread_only = params.get("unread_only", False)
    search = params.get("search", "")
    provider = params.get("provider", "imap")

    imap_config = _get_imap_config(provider)
    if not imap_config:
        return {"success": False, "error": f"IMAP config not found for {provider}"}

    try:
        import email

        mail = imaplib.IMAP4_SSL(imap_config["host"], imap_config["port"])
        mail.login(imap_config["username"], imap_config["password"])
        mail.select(folder)

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
                "body": _extract_email_body(msg),
            })

        mail.close()
        mail.logout()

        return {"success": True, "emails": emails, "count": len(emails)}

    except Exception as e:
        return {"success": False, "error": str(e)}


def _extract_email_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode(errors="ignore")
    else:
        return msg.get_payload(decode=True).decode(errors="ignore")
    return ""


async def email_search(params: dict) -> dict:
    query = params.get("query", "")
    folder = params.get("folder", "INBOX")
    limit = params.get("limit", 20)
    provider = params.get("provider", "imap")

    params_read = {
        "folder": folder,
        "limit": limit,
        "search": query,
        "provider": provider,
    }
    return await email_read(params_read)


# =============================================================================
# NOTIFICATION TOOLS
# =============================================================================

async def notify_desktop(params: dict) -> dict:
    title = params.get("title", "VEGA")
    message = params.get("message", "")
    urgency = params.get("urgency", "normal")
    timeout = params.get("timeout", 5000)
    icon = params.get("icon")

    if not message:
        return {"success": False, "error": "Message required"}

    try:
        import platform
        system = platform.system().lower()

        if system == "linux":
            cmd = ["notify-send", f"--urgency={urgency}", f"--expire-time={timeout}"]
            if icon:
                cmd.extend(["--icon", icon])
            cmd.extend([title, message])
            subprocess.run(cmd)

        elif system == "darwin":
            script = f'display notification "{message}" with title "{title}"'
            if urgency == "critical":
                script += ' sound name "Sosumi"'
            subprocess.run(["osascript", "-e", script])

        elif system == "windows":
            import winrt.windows.ui.notifications as notifications
            import winrt.windows.data.xml.dom as dom

            xml = f"""
            <toast>
                <visual>
                    <binding template="ToastGeneric">
                        <text>{title}</text>
                        <text>{message}</text>
                    </binding>
                </visual>
            </toast>
            """
            doc = dom.XmlDocument()
            doc.load_xml(xml)
            toast = notifications.ToastNotification(doc)
            notifications.ToastNotificationManager.create_toast_notifier("VEGA").show(toast)

        return {"success": True, "message": "Notification sent"}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def notify_push(params: dict) -> dict:
    return {"success": False, "error": "Push notifications not fully implemented"}


# =============================================================================
# CALENDAR TOOLS (placeholder)
# =============================================================================

async def calendar_list(params: dict) -> dict:
    return {"success": False, "error": "Calendar integration not fully implemented"}


async def calendar_create(params: dict) -> dict:
    return {"success": False, "error": "Calendar integration not fully implemented"}


# =============================================================================
# CONTACT TOOLS (placeholder)
# =============================================================================

async def contacts_list(params: dict) -> dict:
    return {"success": False, "error": "Contacts not fully implemented"}


async def contacts_add(params: dict) -> dict:
    return {"success": False, "error": "Contacts not fully implemented"}


# =============================================================================
# HELPERS
# =============================================================================

def _get_smtp_config(provider: str) -> dict | None:
    configs = {
        "gmail": {
            "host": "smtp.gmail.com",
            "port": 587,
            "username": settings.get("gmail_username"),
            "password": settings.get("gmail_app_password"),
        },
        "outlook": {
            "host": "smtp.office365.com",
            "port": 587,
            "username": settings.get("outlook_username"),
            "password": settings.get("outlook_password"),
        },
        "smtp": {
            "host": settings.get("smtp_host"),
            "port": settings.get("smtp_port", 587),
            "username": settings.get("smtp_username"),
            "password": settings.get("smtp_password"),
        },
    }
    return configs.get(provider)


def _get_imap_config(provider: str) -> dict | None:
    configs = {
        "gmail": {
            "host": "imap.gmail.com",
            "port": 993,
            "username": settings.get("gmail_username"),
            "password": settings.get("gmail_app_password"),
        },
        "outlook": {
            "host": "outlook.office365.com",
            "port": 993,
            "username": settings.get("outlook_username"),
            "password": settings.get("outlook_password"),
        },
        "imap": {
            "host": settings.get("imap_host"),
            "port": settings.get("imap_port", 993),
            "username": settings.get("imap_username"),
            "password": settings.get("imap_password"),
        },
    }
    return configs.get(provider)


# =============================================================================
# REGISTRATION
# =============================================================================

def register_communication_tools(registry: ToolRegistry) -> None:
    tools = [
        ("email_send", email_send, "Send email", [
            ToolParameter("to", "array", "Recipient email(s)", required=True),
            ToolParameter("subject", "string", "Email subject", required=True),
            ToolParameter("body", "string", "Email body", required=True),
            ToolParameter("cc", "array", "CC recipients"),
            ToolParameter("bcc", "array", "BCC recipients"),
            ToolParameter("attachments", "array", "Attachment file paths"),
            ToolParameter("html", "boolean", "Body is HTML", default=False),
            ToolParameter("provider", "string", "Provider: gmail, outlook, smtp", default="smtp"),
        ], {"requires_approval": True, "sandboxed": True}),

        ("email_read", email_read, "Read emails", [
            ToolParameter("folder", "string", "Mail folder", default="INBOX"),
            ToolParameter("limit", "integer", "Max emails", default=20),
            ToolParameter("unread_only", "boolean", "Only unread", default=False),
            ToolParameter("search", "string", "Search query"),
            ToolParameter("provider", "string", "Provider: gmail, outlook, imap", default="imap"),
        ], {"requires_approval": False, "sandboxed": True}),

        ("email_search", email_search, "Search emails", [
            ToolParameter("query", "string", "Search query", required=True),
            ToolParameter("folder", "string", "Mail folder", default="INBOX"),
            ToolParameter("limit", "integer", "Max results", default=20),
            ToolParameter("provider", "string", "Provider", default="imap"),
        ], {"requires_approval": False, "sandboxed": True}),

        ("notify_desktop", notify_desktop, "Send desktop notification", [
            ToolParameter("title", "string", "Notification title", default="VEGA"),
            ToolParameter("message", "string", "Notification message", required=True),
            ToolParameter("urgency", "string", "Urgency: low, normal, critical", default="normal"),
            ToolParameter("timeout", "integer", "Timeout ms", default=5000),
            ToolParameter("icon", "string", "Icon path"),
        ], {"requires_approval": False, "sandboxed": False}),

        ("notify_push", notify_push, "Send push notification", [
            ToolParameter("token", "string", "Device token", required=True),
            ToolParameter("title", "string", "Title", required=True),
            ToolParameter("body", "string", "Message body", required=True),
            ToolParameter("data", "object", "Custom data"),
            ToolParameter("provider", "string", "Provider: firebase, onesignal"),
        ], {"requires_approval": True, "sandboxed": True}),

        ("calendar_list", calendar_list, "List calendar events", [], {"requires_approval": False, "sandboxed": True}),

        ("calendar_create", calendar_create, "Create calendar event", [], {"requires_approval": True, "sandboxed": True}),

        ("contacts_list", contacts_list, "List contacts", [], {"requires_approval": False, "sandboxed": True}),

        ("contacts_add", contacts_add, "Add contact", [], {"requires_approval": True, "sandboxed": True}),
    ]

    for name, func, desc, params, extra in tools:
        metadata = ToolMetadata(
            name=name,
            description=desc,
            category="communication",
            parameters=params,
            returns={"type": "object"},
            sandboxed=extra.get("sandboxed", True),
            requires_approval=extra.get("requires_approval", False),
            tags=["communication", "email", "notification", "calendar"],
        )
        registry.register(name, func, metadata)

    logger.info("Registered communication tools")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "register_communication_tools",
]