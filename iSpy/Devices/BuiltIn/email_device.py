import logging
import smtplib
from email.mime.text import MIMEText

from iSpy.Devices.device import Device


class EmailDevice(Device):
    def __init__(
        self,
        name: str = "Gmail",
        sender: str = "",
        app_password: str = "",
        recipients: list[str] | None = None,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
    ):
        super().__init__(name)
        self.sender = sender
        self.app_password = app_password
        self.recipients = recipients or []
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    def verify(self) -> bool:
        if not self.sender or not self.app_password or not self.recipients:
            self.logger.warning(
                "Email device needs sender, app_password, and recipients."
            )
            return False
        try:
            with self._connect():
                self.logger.info(
                    "SMTP login ok - %s can receive alerts.", ", ".join(self.recipients)
                )
            return True
        except Exception as e:
            self.logger.warning("SMTP verification failed: %s", e)
            return False

    def notify(self, message: str, **payload) -> bool:
        try:
            body = message
            if payload:
                body += "\n\n" + "\n".join(f"{k}: {v}" for k, v in payload.items())
            msg = MIMEText(body)
            msg["Subject"] = f"[iSpy] {message[:60]}"
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.recipients)
            with self._connect() as smtp:
                smtp.send_message(msg)
            self.logger.info("Email sent to %s", ", ".join(self.recipients))
            return True
        except Exception as e:
            self.logger.error("Email failed: %s", e)
            return False

    def _connect(self):
        smtp = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15)
        smtp.starttls()
        smtp.login(self.sender, self.app_password)
        return smtp