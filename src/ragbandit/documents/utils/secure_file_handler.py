"""Utilities for secure file handling with encryption."""

import os
import tempfile
from pathlib import Path
from cryptography.fernet import Fernet, InvalidToken
import shutil


class SecureFileHandler:
    """Handles secure file operations with encryption at rest."""

    def __init__(self, encryption_key: str):
        """Initialize the secure file handler with an encryption key.

        Args:
            encryption_key: The encryption key to use for file operations

        Raises:
            ValueError: If encryption_key is empty or invalid format
        """
        if not encryption_key:
            raise ValueError("Encryption key cannot be empty")
        try:
            # Validate the key by creating a cipher
            self._cipher = Fernet(encryption_key.encode())
        except InvalidToken:
            raise ValueError("Invalid encryption key format")

    def save_encrypted_file(
        self, content: bytes, prefix: str = "doc", original_file_name=""
    ) -> Path:
        """Save file content with encryption.

        Args:
            content: Raw bytes to encrypt and save
            prefix: Prefix for the temporary file name

        Returns:
            Path to the encrypted file
        """
        # Create a temporary directory that only this process can access
        temp_dir = Path(tempfile.mkdtemp(prefix="secure_"))
        try:
            # Create encrypted file path
            suffix = ""
            if original_file_name:
                suffix = Path(original_file_name).suffix

            file_path = temp_dir / f"{prefix}_{os.urandom(8).hex()}{suffix}"

            # Encrypt and save
            encrypted_content = self._cipher.encrypt(content)
            file_path.write_bytes(encrypted_content)

            return file_path

        except Exception as e:
            # Clean up on error
            shutil.rmtree(temp_dir)
            raise e

    def read_encrypted_file(self, file_path: Path) -> bytes:
        """Read and decrypt file content.

        Args:
            file_path: Path to the encrypted file

        Returns:
            Decrypted content as bytes
        """
        encrypted_content = file_path.read_bytes()
        return self._cipher.decrypt(encrypted_content)

    def secure_delete(self, file_path: Path):
        """Securely delete a file and its parent directory.

        Args:
            file_path: Path to the file to delete
        """
        if file_path.exists():
            try:
                # Securely overwrite file contents before deletion
                # Write random data 3 times to make recovery harder
                file_size = file_path.stat().st_size
                for _ in range(3):
                    with open(file_path, "wb") as f:
                        f.write(os.urandom(file_size))
                        f.flush()
                        os.fsync(f.fileno())

                # Now delete the parent directory and all its contents
                shutil.rmtree(file_path.parent)
            except FileNotFoundError:
                pass  # Already deleted
