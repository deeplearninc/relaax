from __future__ import print_function

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
    def __init__(self, bucket, key, aws_access_key=None, aws_secret_key=None):
        super(S3Saver, self).__init__()
        self._bucket = bucket
        self._key = key
        self._aws_access_key = aws_access_key
        self._aws_secret_key = aws_secret_key

    def global_steps(self):
        steps = set()
        for name in self._list_objects('%s-' % self._CHECKPOINT_PREFIX):
            match = re.match('^%s-(\d+)(?:|\..+)$' % self._CHECKPOINT_PREFIX, name)
            if match is not None:
                steps.add(int(match.group(1)))
        return steps

    def remove_checkpoint(self, global_step):
        removed = False
        for name in self._list_objects('%s-%d' % (self._CHECKPOINT_PREFIX, global_step)):
            match = re.match('^%s-\d+(?:|\..+)$' % self._CHECKPOINT_PREFIX, name)
            if match is not None:
                self._remove(name)
                removed = True
        if removed:
            _logger.info('checkpoint {} was removed from {} bucket {} key'.format(global_step, self._bucket, self._key))

    def restore_checkpoint(self, session, global_step):
        with _temp_dir() as dir:
            for name in self._list_objects('%s-%d' % (self._CHECKPOINT_PREFIX, global_step)):
                match = re.match('^%s-\d+(?:|\..+)$' % self._CHECKPOINT_PREFIX, name)
                if match is not None:
                    self._download(dir, name)
            tensorflow.train.Saver().restore(
                session,
                os.path.join(dir, '%s-%d' % (self._CHECKPOINT_PREFIX, global_step))
            )
        _logger.info('checkpoint {} was restored from {} bucket {} key'.format(global_step, self._bucket, self._key))

    def save_checkpoint(self, session, global_step):
        with _temp_dir() as dir:
            tensorflow.train.Saver().save(
                session,
                '%s/%s' % (dir, self._CHECKPOINT_PREFIX),
                global_step=global_step
            )

            for name in self._list_entries(dir, '%s-%d' % (self._CHECKPOINT_PREFIX, global_step)):
                match = re.match('^%s-\d+(?:|\..+)$' % self._CHECKPOINT_PREFIX, name)
                if match is not None:
                    self._upload(dir, name)
        _logger.info('checkpoint {} was saved to {} bucket {} key'.format(global_step, self._bucket, self._key))

    def _list_entries(self, dir, prefix):
        for name in os.listdir(dir):
            if (
                (name == prefix or name.startswith('%s.' % prefix)) and
                os.path.isfile(os.path.join(dir, name))
            ):
                yield name

    def _list_objects(self, prefix):
        full_prefix = '%s/%s' % (self._key, prefix)
        for obj in self._s3().Bucket(self._bucket).objects.filter(Prefix=full_prefix):
            assert obj.key.startswith(full_prefix)
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
