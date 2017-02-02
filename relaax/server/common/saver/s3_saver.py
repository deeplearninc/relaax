from __future__ import print_function

import boto3
import botocore
import contextlib
import os
import re
import shutil
import tempfile
import tensorflow

import saver


class S3Saver(saver.Saver):
    def __init__(self, bucket, key, aws_access_key=None, aws_secret_key=None):
        super(S3Saver, self).__init__()
        self._bucket = bucket
        self._key = key
        self._aws_access_key = aws_access_key
        self._aws_secret_key = aws_secret_key

    def restore_latest_checkpoint(self, session):
        with _temp_dir() as dir:
            lf_path = self._download(dir, self._LATEST_FILENAME)
            if lf_path is None:
                return False
            cp_name = self._latest_cp_name(dir)
            for name in self._list_keys(cp_name):
                self._download(dir, name)
            cp_path = os.path.join(dir, cp_name)
            self._get_saver().restore(session, cp_path)
        return True

    def save_checkpoint(self, session, global_step):
        with _temp_dir() as dir:
            self._save(dir, session, global_step)

            cp_name = self._latest_cp_name(dir)
            if cp_name.startswith('%s/' % dir):
                cp_name = cp_name[len(dir) + 1:]
                with open('%s/%s' % (dir, self._LATEST_FILENAME), 'w') as f:
                    print('model_checkpoint_path: "%s"' % cp_name, file=f)
                    print('all_model_checkpoint_paths: "%s"' % cp_name, file=f)

            cp_name = self._latest_cp_name(dir)
            for f in self._list_entries(dir, cp_name):
                self._upload(dir, f)
            self._upload(dir, self._LATEST_FILENAME)

    def location(self):
        return "'%s' bucket '%s' key" % (self._bucket, self._key)

    def _latest_cp_name(self, dir):
        cp_name = None
        with open('%s/%s' % (dir, self._LATEST_FILENAME), 'r') as f:
            for line in f:
                match = re.match('^model_checkpoint_path:\s*"(.*)"\s*$', line)
                if match is not None:
                    cp_name = match.group(1)
        return cp_name

    def _list_entries(self, dir, prefix):
        for name in os.listdir(dir):
            if (
                (name == prefix or name.startswith('%s.' % prefix)) and
                os.path.isfile(os.path.join(dir, name))
            ):
                yield name

    def _list_keys(self, prefix):
        for obj in self._s3().Bucket(self._bucket).objects.filter(Prefix='%s/%s' % (self._key, prefix)):
            assert obj.key.startswith('%s/' % self._key)
            name = obj.key[len(self._key) + 1:]
            if name == prefix or name.startswith('%s.' % prefix):
                yield name

    def _download(self, dir, name):
        if not self._bucket_exists():
            return None
        path = '%s/%s' % (dir, name)
        try:
            self._s3().meta.client.download_file(
                self._bucket,
                '%s/%s' % (self._key, name),
                path
            )
        except botocore.exceptions.ClientError as e:
            if int(e.response['Error']['Code']) == 404:
                return None 
            raise
        return path

    def _upload(self, dir, name):
        if not self._bucket_exists():
            self._create_bucket()
        self._s3().meta.client.upload_file(
            '%s/%s' % (dir, name),
            self._bucket,
            '%s/%s' % (self._key, name)
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
