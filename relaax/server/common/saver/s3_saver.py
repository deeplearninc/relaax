import boto3
import botocore
import contextlib
import logging
import os
import re
import shutil
import tempfile
import tensorflow

import saver


_logger = logging.getLogger(__name__)


class S3Saver(saver.Saver):
    def __init__(self, checkpoint, bucket_key, aws_access_key=None, aws_secret_key=None):
        super(S3Saver, self).__init__()
        self._checkpoint = checkpoint
        match = re.match('^([^/]+)/(.+)$', bucket_key)
        self._bucket = match.group(1)
        self._key = match.group(2)
        self._aws_access_key = aws_access_key
        self._aws_secret_key = aws_secret_key

    def checkpoint_ids(self):
        return self._checkpoint.checkpoint_ids(self._listdir())

    def remove_checkpoint(self, checkpoint_id):
        removed = False
        for name in self._checkpoint.checkpoint_names(self._listdir(), checkpoint_id):
            self._remove(name)
            removed = True
        if removed:
            _logger.info('checkpoint {} was removed from {} bucket {} key'.format(checkpoint_id, self._bucket, self._key))

    def restore_checkpoint(self, checkpoint_id):
        with _temp_dir() as dir:
            for name in self._checkpoint.checkpoint_names(self._listdir(), checkpoint_id):
                self._download(dir, name)
            self._checkpoint.restore_checkpoint(dir, checkpoint_id)
        _logger.info('checkpoint {} was restored from {} bucket {} key'.format(checkpoint_id, self._bucket, self._key))

    def save_checkpoint(self, checkpoint_id):
        with _temp_dir() as dir:
            self._checkpoint.save_checkpoint(dir, checkpoint_id)
            for name in os.listdir(dir):
                self._upload(dir, name)
        _logger.info('checkpoint {} was saved to {} bucket {} key'.format(checkpoint_id, self._bucket, self._key))

    def _listdir(self):
        prefix = '%s/' % self._key
        for obj in self._s3().Bucket(self._bucket).objects.filter(Prefix=prefix):
            assert obj.key.startswith(prefix)
            yield obj.key[len(self._key) + 1:]

    def _download(self, dir, name):
        if not self._bucket_exists():
            return
        path = '%s/%s' % (dir, name)
        try:
            self._s3().meta.client.download_file(
                Bucket=self._bucket,
                Key='%s/%s' % (self._key, name),
                Filename=path
            )
        except botocore.exceptions.ClientError as e:
            if int(e.response['Error']['Code']) == 404:
                return
            raise

    def _upload(self, dir, name):
        if not self._bucket_exists():
            self._create_bucket()
        self._s3().meta.client.upload_file(
            Filename='%s/%s' % (dir, name),
            Bucket=self._bucket,
            Key='%s/%s' % (self._key, name)
        )

    def _remove(self, name):
        self._s3().meta.client.delete_object(
            Bucket=self._bucket,
            Key='%s/%s' % (self._key, name)
        )

    def _bucket_exists(self):
        s3 = self._s3()
        try:
            s3.meta.client.head_bucket(Bucket=self._bucket)
        except botocore.exceptions.ClientError as e:
            # If a client error is thrown, then check that it was a 404 error.
            # If it was a 404 error, then the bucket does not exist.
            if int(e.response['Error']['Code']) == 404:
                return False 
        return True

    def _create_bucket(self):
        self._s3().create_bucket(Bucket=self._bucket)

    def _s3(self):
        return self._session().resource('s3')

    def _session(self):
        return boto3.session.Session(
            aws_access_key_id=self._aws_access_key,
            aws_secret_access_key=self._aws_secret_key
        )


@contextlib.contextmanager
def _temp_dir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)
