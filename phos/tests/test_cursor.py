import unittest
from phos import Phos, FetchWithoutExecuteError, NoResultSetError
from phos.cursor import Query
import pyhive
from mock import MagicMock, patch, PropertyMock


class TestCursor(unittest.TestCase):
    """
    Tests Cursor override methods of Phos.
    """

    def test_fetchall_requires_exec_first(self):
        c = Phos(MagicMock())
        with self.assertRaises(FetchWithoutExecuteError):
            c.fetchall()

    def test_description_requires_exec_first(self):
        c = Phos(MagicMock())
        with self.assertRaises(FetchWithoutExecuteError):
            c.description

    @patch('pyhive.hive.Cursor.execute')
    def test_execute_query_no_cache_calls_super(self, super_exec):
        c = Phos(MagicMock())
        q = 'Hallo'
        c.execute(q, _cache=False)
        super_exec.assert_called_with(q)

    @patch('pyhive.hive.Cursor.execute')
    @patch.object(Query, 'cache_exists', return_value=False)
    def test_execute_query_cache_no_exists_calls_super(self, cache_exists, super_exec):
        c = Phos(MagicMock())

        q = 'Hallo'
        c.execute(q, _cache=True)
        super_exec.assert_called_with(q)

    @patch('pyhive.hive.Cursor.execute')
    @patch.object(Query, 'cache_exists', return_value=True)
    def test_execute_query_cache_exists_does_not_call_super(self, cache_exists, super_exec):
        c = Phos(MagicMock())

        q = 'Hallo'
        c.execute(q, _cache=True)
        self.assertFalse(super_exec.called, "Super.execute should not have been called when cache is " +
                                            "demanded and exists.")

    @patch('pyhive.hive.Cursor.fetchall')
    @patch('pyhive.hive.Cursor.execute')
    def test_fetchall_no_cache_calls_super(self, super_exec, super_fetch):
        c = Phos(MagicMock())
        q = 'Hallo'
        c.execute(q, _cache=False)
        c.fetchall()
        super_fetch.assert_called()

    @patch('pyhive.hive.Cursor.description', new_callable=PropertyMock)
    @patch('pyhive.hive.Cursor.execute')
    def test_description_no_cache_calls_super(self, super_exec, super_desc):
        c = Phos(MagicMock())
        q = 'Hallo'
        c.execute(q, _cache=False)
        c.description()
        super_desc.assert_called()

    @patch.object(pyhive.hive.Cursor, 'fetchall', return_value='Some data')
    @patch('pyhive.hive.Cursor.execute')
    def test_fetchall_cache_not_exists_dumps_and_calls_super(self, super_exec, super_fetch):
        c = Phos(MagicMock())
        q = 'Hallo'
        c.execute(q, _cache=True, _recache=True)
        c.del_cache()
        dumped_and_loaded = c.fetchall()
        self.assertEqual(dumped_and_loaded, 'Some data',
                         "Data dumped loaded from cache does not match what super.fetchall returned.")
        self.assertTrue(super_fetch.called)

    @patch.object(pyhive.hive.Cursor, 'description', new_callable=PropertyMock, return_value='Some data')
    @patch('pyhive.hive.Cursor.execute')
    def test_description_cache_not_exists_dumps_and_calls_super(self, super_exec, super_desc):
        c = Phos(MagicMock())
        q = 'Hallo'
        c.execute(q, _cache=True, _recache=True)
        c.del_cache()
        dumped_and_loaded = c.description
        self.assertEqual(dumped_and_loaded, 'Some data',
                         "Data dumped loaded from cache does not match what super.description returned.")
        self.assertTrue(super_desc.called)

    @patch('pyhive.hive.Cursor.fetchall')
    @patch('pyhive.hive.Cursor.execute')
    def test_fetchall_cache_exists_does_not_dump_and_does_not_call_super(self, super_exec, super_fetch):
        c = Phos(MagicMock())
        q = 'Hallo'
        c.execute(q, _cache=True, _recache=True)
        c.del_cache()
        cached = 'Some data'
        c.q.write_cache({'data': cached})

        dumped_and_loaded = c.fetchall()
        self.assertEqual(dumped_and_loaded, cached)
        self.assertFalse(super_fetch.called)

    @patch('pyhive.hive.Cursor.description', new_callable=PropertyMock)
    @patch('pyhive.hive.Cursor.execute')
    def test_description_cache_exists_does_not_dump_and_does_not_call_super(self, super_exec, super_desc):
        c = Phos(MagicMock())
        q = 'Hallo'
        c.execute(q, _cache=True, _recache=True)
        c.del_cache()
        cached = 'Some data'
        c.q.write_cache({'data': 'who cares', 'description': cached})

        dumped_and_loaded = c.description
        self.assertEqual(dumped_and_loaded, cached)
        self.assertFalse(super_desc.called)

    def test_execute_progress_is_called_when_required(self):
        """
            Makes sure the async_ args is passed to execute, no matter what the other params are.
            Stops the execution after the call to execute because it is irrelevant.
        """
        class CallException(Exception):
            pass

        with patch.object(pyhive.hive.Cursor, 'execute', side_effect=CallException("Execute called")) as super_exec:
            c = Phos(MagicMock())

            try:
                c.execute('Hallo', _cache=False, _progress=True)
            except CallException:
                args, kwargs = super_exec.call_args_list[0]
                self.assertTrue(kwargs.get('async_', False))
            else:
                self.fail("super.execute should have been called when _cache=False")

    def test_execute_progress_is_not_called_when_not_required(self):
        """
            Makes sure the async_ args is passed to execute, no matter what the other params are.
            Stops the execution after the call to execute because it is irrelevant.
        """
        class CallException(Exception):
            pass

        with patch.object(pyhive.hive.Cursor, 'execute', side_effect=CallException("Execute called")) as super_exec:
            c = Phos(MagicMock())

            try:
                c.execute('Hallo', _cache=False, _progress=False)
            except CallException:
                args, kwargs = super_exec.call_args_list[0]
                self.assertFalse(kwargs.get('async_', False))
            else:
                self.fail("super.execute should have been called when _cache=False")
