from __future__ import print_function

import boto3
import botocore
import contextlib
import os
import shutil
import tempfile
import tensorflow

import saver


class S3Saver(object):
    def __init__(self, bucket, key, aws_access_key=None, aws_secret_key=None):
        super(S3Saver, self).__init__()
        self._bucket = bucket
        self._key = key
        self._aws_access_key = aws_access_key
        self._aws_secret_key = aws_secret_key

    def restore_latest_checkpoint(self, session):
        with _temp_dir() as dir:
            lf_path = self._download(dir, _LATEST_FILENAME)
            if lf_path is None:
                return False
            cp_path = self._download(dir, self._latest_cp_name(dir))
            if cp_path is None:
                return False
            self._saver.restore(session, cp_path)
        return True

    def save_checkpoint(self, session, global_step):
        with _temp_dir() as dir:
            self._save(dir, session, global_step)
            self._upload(dir, self._latest_cp_name(dir))
            self._upload(dir, _LATEST_FILENAME)

    def _latest_cp_name(self, dir):
        return os.path.basename(self._latest_cp_path(dir))

    def _download(self, dir, name):
        if not self._bucket_exists():
            return None
        self._s3().download_file(
            self._bucket,
            '%s/%s' % (self._key, name),
            '%s/%s' % (dir, name)
        )

    def _upload(self, dir, name):
        if not self._bucket_exists():
            self._create_bucket()
        self._s3().upload_file(
            '%s/%s' % (dir, name),
            self._bucket,
            '%s/%s' % (self._key, name)
        )

    def _bucket_exists(self):
        session = self._session()
        try:
            session.meta.client.head_bucket(Bucket='mybucket')
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
