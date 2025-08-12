import logging
import io


class InMemoryLogHandler(logging.Handler):
    """
    Collects every formatted log record that flows through it
    into an in-memory buffer.  Thread-safe because `logging`
    already locks `emit()`.
    """
    def __init__(self, level=logging.INFO,
                 fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s"):
        super().__init__(level)
        self.buffer = io.StringIO()
        self.setFormatter(logging.Formatter(fmt))
        print("InMemoryLogHandler initialized with level:", level)
        print("Current root logger level:", logging.getLogger().level)

    def emit(self, record):
        print(
            f"Emit called with record: {record.levelname} - "
            f"{record.name} - {record.msg}"
        )
        print(f"Record level: {record.levelno}, Handler level: {self.level}")
        self.buffer.write(self.format(record) + "\n")

    def dump(self) -> str:
        """Return the whole transcript so far."""
        return self.buffer.getvalue()

    def clear(self) -> None:
        self.buffer.truncate(0)
        self.buffer.seek(0)
