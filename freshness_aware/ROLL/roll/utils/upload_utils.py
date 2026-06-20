import contextlib
import os
import shutil

from filelock import FileLock

from roll.utils.logging import get_logger


logger = get_logger()

uploader_registry = {}


class FileSystemUploader:
    """
    将本地的ckpt目录上传到文件系统, oss/cpfs在多Role的场景下，
    每个Role会把自己的ckpt dir的内容上传到OUTPUT_DIR/ckpt_id/下
    {
        "type": "file_system",
        "output_dir": /data/oss_bucket_0/llm/models
    }
    """

    def __init__(self, output_dir, keep_local_file=False, *args, **kwargs):
        self.output_dir = output_dir
        self.keep_local_file = keep_local_file
        logger.info(f"use FileSystemUploader to upload {output_dir}")

    @staticmethod
    def _same_filesystem(src_path: str, dst_dir: str) -> bool:
        try:
            return os.stat(src_path).st_dev == os.stat(dst_dir).st_dev
        except OSError:
            return False

    @classmethod
    def _merge_move(cls, src_dir: str, dst_dir: str):
        os.makedirs(dst_dir, exist_ok=True)
        for name in os.listdir(src_dir):
            src_item = os.path.join(src_dir, name)
            dst_item = os.path.join(dst_dir, name)
            if os.path.isdir(src_item) and os.path.isdir(dst_item):
                cls._merge_move(src_item, dst_item)
                with contextlib.suppress(OSError):
                    os.rmdir(src_item)
                continue
            if os.path.exists(dst_item):
                if os.path.isdir(dst_item):
                    shutil.rmtree(dst_item)
                else:
                    os.remove(dst_item)
            shutil.move(src_item, dst_item)

    def upload(self, ckpt_id: str, local_state_path: str, keep_local_file=None, **kwargs):
        ckpt_id_output_dir = os.path.join(self.output_dir, ckpt_id)
        os.makedirs(ckpt_id_output_dir, exist_ok=True)
        logger.info(f"{local_state_path} save to {ckpt_id_output_dir}, wait...")
        keep_local_file = self.keep_local_file if keep_local_file is None else keep_local_file
        lock_path = os.path.join(ckpt_id_output_dir, ".upload.lock")
        with FileLock(lock_path):
            if (
                not keep_local_file
                and os.path.isdir(local_state_path)
                and self._same_filesystem(local_state_path, ckpt_id_output_dir)
            ):
                logger.info(f"{local_state_path} move-merge to {ckpt_id_output_dir}")
                self._merge_move(local_state_path, ckpt_id_output_dir)
            else:
                shutil.copytree(local_state_path, ckpt_id_output_dir, dirs_exist_ok=True)
        logger.info(f"{local_state_path} save to {ckpt_id_output_dir}, done...")


uploader_registry['file_system'] = FileSystemUploader
