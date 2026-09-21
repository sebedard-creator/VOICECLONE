import hashlib
import json
import tempfile
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from colab.checkpoint_store import CheckpointStore
from core.drive_archive import DriveArchive, VoiceCloneError, digest, safe_component
from core.drive_sync import GoogleDriveClient


class Request:
    def __init__(self, result):
        self.result = result

    def execute(self, **kwargs):
        return self.result() if callable(self.result) else self.result


class FakeFiles:
    def __init__(self, entries):
        self.entries = entries
        self.deleted = []

    def get(self, fileId, **kwargs):
        return Request(lambda: dict(self.entries[fileId]))

    def delete(self, fileId, **kwargs):
        def run():
            self.deleted.append(fileId)
            del self.entries[fileId]
            return {}
        return Request(run)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        content = b'voice data'
        checksum = hashlib.md5(content).hexdigest()
        self.entries = {str(i): dict(id=str(i), name=f'voice{i}.pth', mimeType='binary',
                                    parents=['root-id'], size=str(len(content)), md5Checksum=checksum,
                                    version='1', modifiedTime='now', parts=['RVC_Output', f'voice{i}.pth'])
                        for i in range(2)}
        self.files = FakeFiles(self.entries)
        self.downloads = []

        def download(fid, path):
            self.downloads.append(fid)
            path.write_bytes(content)
            return path
        client = SimpleNamespace(service=SimpleNamespace(files=lambda: self.files), download_file=download)
        self.archive = DriveArchive(SimpleNamespace(root_path=self.root), client)
        self.archive.roots = lambda: [{'id': 'root-id', 'name': 'RVC_Output'}]
        self.archive.inventory = lambda roots=None: [dict(self.entries[k]) for k in sorted(self.entries)]

    def test_archive_deduplicates_and_reuses_verified_blobs(self):
        path = self.archive.snapshot()
        data = self.archive.verify(path)
        self.assertEqual(len(self.downloads), 1)
        self.assertEqual(len({f['blob'] for f in data['files']}), 1)
        self.archive.snapshot()
        self.assertEqual(len(self.downloads), 1)

    def test_corrupt_archive_blocks_all_deletion(self):
        path = self.archive.snapshot()
        blob = next((self.archive.root / 'objects').iterdir())
        blob.write_bytes(b'corrupt')
        with self.assertRaises(VoiceCloneError):
            self.archive.remove_archived(path, True)
        self.assertEqual(self.files.deleted, [])

    def test_changed_remote_blocks_all_deletion(self):
        path = self.archive.snapshot()
        self.entries['1']['version'] = '2'
        with self.assertRaises(VoiceCloneError):
            self.archive.remove_archived(path, True)
        self.assertEqual(self.files.deleted, [])

    def test_new_remote_file_blocks_all_deletion(self):
        path = self.archive.snapshot()
        self.entries['2'] = dict(self.entries['1'], id='2')
        with self.assertRaises(VoiceCloneError):
            self.archive.remove_archived(path, True)
        self.assertEqual(self.files.deleted, [])

    def test_cleanup_requires_explicit_idle_confirmation(self):
        path = self.archive.snapshot()
        with self.assertRaises(VoiceCloneError):
            self.archive.remove_archived(path)
        self.assertEqual(self.files.deleted, [])

    def test_cleanup_retains_verified_archive_and_retries(self):
        path = self.archive.snapshot()
        self.assertEqual(self.archive.remove_archived(path, True), 20)
        self.assertEqual(self.archive.remove_archived(path, True), 0)
        self.assertEqual(len(self.archive.verify(path)['files']), 2)

    def test_change_during_archive_prevents_complete_snapshot(self):
        original = self.archive.client.download_file
        def download(fid, path):
            result = original(fid, path)
            self.entries[fid]['version'] = '2'
            return result
        self.archive.client.download_file = download
        with self.assertRaises(VoiceCloneError):
            self.archive.snapshot()
        self.assertFalse((self.archive.root / 'snapshots').exists())

    def test_download_corruption_is_not_committed(self):
        self.archive.client.download_file = lambda fid, path: path.write_bytes(b'bad')
        with self.assertRaises(VoiceCloneError):
            self.archive.snapshot()
        self.assertEqual(list((self.archive.root / 'objects').iterdir()), [])

    def test_restore_requires_training_weights(self):
        path = self.archive.snapshot()
        with self.assertRaises(VoiceCloneError):
            self.archive.restore_actor(path, 'voice')

    def test_unsafe_names_refused(self):
        for name in ('../outside', 'C:\\secret', 'a/b', 'a:b', 'CON', 'x.', '..'):
            with self.subTest(name=name), self.assertRaises(VoiceCloneError):
                safe_component(name)

    def test_restore_uploads_training_only_and_preserves_names(self):
        self.entries['0'].update(name='G_1.pth', parts=['VOICECLONE_Training', 'Lily', 'checkpoints', 'G_1.pth'])
        self.entries['1'].update(name='D_1.pth', parts=['VOICECLONE_Training', 'Lily', 'checkpoints', 'D_1.pth'])
        self.entries['2'] = dict(self.entries['1'], id='2', name='Lily_dataset_cleaned.zip', parts=['VOICECLONE_Datasets', 'Lily_dataset_cleaned.zip'])
        path = self.archive.snapshot()
        self.archive.roots = lambda: [{'name': 'VOICECLONE_Training', 'id': 'training'}]
        self.archive.client._escape_query_literal = GoogleDriveClient._escape_query_literal
        made = []
        def create(body, **kwargs):
            result = dict(body, id=str(len(made)))
            made.append(result)
            return Request(result)
        self.files.create = create
        uploaded = []
        self.archive.client.upload_file = lambda path, folder, name: uploaded.append((name, folder, path.read_bytes()))
        with patch('core.drive_archive.list_files', return_value=[]):
            count = self.archive.restore_actor(path, 'Lily')
        self.assertEqual(count, 2)
        self.assertEqual({item[0] for item in uploaded}, {'G_1.pth', 'D_1.pth'})
        self.assertTrue(all(item[2] == b'voice data' for item in uploaded))
        self.assertEqual([folder['name'] for folder in made], ['Lily', 'checkpoints'])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.local = self.root / 'local'
        self.local.mkdir()
        self.store = CheckpointStore(self.root / 'drive', epoch_reader=lambda p: int(p.read_text()))

    def pair(self, folder, g, d=None):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'G_2333333.pth').write_text(str(g))
        (folder / 'D_2333333.pth').write_text(str(g if d is None else d))

    def test_one_pair_retained_and_data_not_duplicated(self):
        self.pair(self.local, 10)
        (self.local / 'features.npy').write_bytes(b'features')
        self.store.save_data(self.local)
        self.assertFalse(list(self.store.experiment.glob('*.pth')))
        self.assertTrue(self.store.save_checkpoints(self.local))
        self.pair(self.local, 200)
        self.assertTrue(self.store.save_checkpoints(self.local))
        self.assertEqual(len(list(self.store.checkpoints.glob('generation_*'))), 1)
        self.assertEqual(len(list(self.store.root.rglob('*.pth'))), 2)
        target = self.root / 'restore'
        self.assertEqual(self.store.restore(target), 200)
        self.assertEqual((target / 'features.npy').read_bytes(), b'features')

    def test_incoherent_pair_keeps_previous_backup(self):
        self.pair(self.local, 10)
        self.store.save_data(self.local)
        self.store.save_checkpoints(self.local)
        before = self.store.pointer.read_bytes()
        self.pair(self.local, 20, 10)
        self.assertFalse(self.store.save_checkpoints(self.local))
        self.assertEqual(self.store.pointer.read_bytes(), before)
        self.assertEqual(self.store.restore(self.root / 'restore'), 10)

    def test_interrupted_upload_preserves_committed_pair(self):
        self.pair(self.local, 10)
        self.store.save_data(self.local)
        self.store.save_checkpoints(self.local)
        before = self.store.pointer.read_bytes()
        self.pair(self.local, 200)
        with patch('colab.checkpoint_store.copy_verified', side_effect=OSError('Drive full')):
            with self.assertRaises(OSError):
                self.store.save_checkpoints(self.local)
        self.assertEqual(self.store.pointer.read_bytes(), before)
        self.assertEqual(self.store.restore(self.root / 'restore'), 10)

    def test_legacy_resume_uses_newest_complete_epoch(self):
        self.pair(self.store.experiment, 10)
        self.pair(self.store.checkpoints, 20)
        self.assertEqual(self.store.restore(self.root / 'restore'), 20)

    def test_legacy_incoherent_newer_pair_falls_back(self):
        self.pair(self.store.experiment, 10)
        self.pair(self.store.checkpoints, 20, 10)
        self.assertEqual(self.store.restore(self.root / 'restore'), 10)

    def test_missing_pair_refuses_silent_restart(self):
        self.store.experiment.mkdir(parents=True)
        with self.assertRaises(RuntimeError):
            self.store.restore(self.root / 'restore')

    def test_corrupt_checkpoint_refuses_resume(self):
        self.pair(self.local, 10)
        self.store.save_data(self.local)
        self.store.save_checkpoints(self.local)
        next(self.store.checkpoints.rglob('G_*.pth')).write_text('999')
        with self.assertRaises(RuntimeError):
            self.store.restore(self.root / 'restore')

    def test_corrupt_features_refuses_resume(self):
        self.pair(self.local, 10)
        (self.local / 'features.npy').write_bytes(b'valid')
        self.store.save_data(self.local)
        self.store.save_checkpoints(self.local)
        (self.store.experiment / 'features.npy').write_bytes(b'bad')
        with self.assertRaises(RuntimeError):
            self.store.restore(self.root / 'restore')


class UploadTests(unittest.TestCase):
    def test_identical_upload_skipped_different_content_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'dataset.zip'
            path.write_bytes(b'dataset')
            client = GoogleDriveClient(None, 'upload', 'output', 'RVC_Output')
            existing = dict(id='existing', md5Checksum=digest(path), size=str(path.stat().st_size))
            with patch('core.drive_archive.list_files', return_value=[existing]):
                self.assertEqual(client.upload_file(path, 'upload')['id'], 'existing')
                path.write_bytes(b'changed')
                with self.assertRaises(VoiceCloneError):
                    client.upload_file(path, 'upload')

    def test_download_verifies_before_replacing_existing_local_file(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'voice.pth'
            destination.write_bytes(b'old model')
            metadata = dict(id='id', name='voice.pth', parents=['folder'], size='5',
                            md5Checksum=hashlib.md5(b'valid').hexdigest(), version='1', modifiedTime='now')
            files = SimpleNamespace(get=lambda **kwargs: Request(metadata), get_media=lambda **kwargs: None)
            client = GoogleDriveClient(SimpleNamespace(files=lambda: files), 'upload', 'output', 'RVC_Output')
            class Downloader:
                payload = b'wrong'
                def __init__(self, handle, request, **kwargs):
                    self.handle = handle
                def next_chunk(self, **kwargs):
                    self.handle.write(self.payload)
                    return None, True
            with patch('googleapiclient.http.MediaIoBaseDownload', Downloader):
                with self.assertRaises(VoiceCloneError):
                    client.download_file('id', destination)
                self.assertEqual(destination.read_bytes(), b'old model')
                self.assertEqual(list(Path(temp).glob('*.part')), [])
                Downloader.payload = b'valid'
                client.download_file('id', destination)
                self.assertEqual(destination.read_bytes(), b'valid')


if __name__ == '__main__':
    unittest.main()
