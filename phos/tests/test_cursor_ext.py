import unittest
from phos import Phos
from phos.cursor import UnknownPlotTypeError
import pyhive
from mock import MagicMock, patch, PropertyMock
import textwrap


# todo: make (part of) this file test_with_data to have a testsetup
class TestCursor(unittest.TestCase):
    """
    Tests extra methods not part of Cursor base class.
    """
    @patch('pyhive.hive.Cursor.execute')
    def test_set_dag_name_generates_proper_query(self, super_exec):
        c = Phos(MagicMock())
        dagname = 'tst dag with UPPER and utf8 ✓'
        c.set_dag_name(dagname)
        arg = super_exec.call_args[0]
        # I don't want to change my tests if case or spaces are changed, so fuzzy testing ftw.
        self.assertTrue('sethive.query.name=' in arg[0].replace(' ', ''))
        self.assertTrue(dagname in arg[0])

    @patch.object(pyhive.hive.Cursor, 'fetchall', return_value=[
        [1, 42, 'str2', 3.5],
        [1, 1234567891234, 'str1', 4.45643245]])
    @patch('phos.Phos.description', new_callable=PropertyMock)
    @patch('pyhive.hive.Cursor.execute')
    def test_pformat(self, super_exec, description, fetchall):
        """
        Overrides description and fetchall (part of Python DP API) to check output.
        """
        desc = [
            ('int', 'int_type'),
            ('bigint', 'bigint_type'),
            ('str', 'string_type'),
            ('float', 'decimal_type')
        ]
        description.return_value = desc
        c = Phos(MagicMock())
        c.execute('exec needs to be called before fetchall', _cache=False)
        # Unsorted
        formatted = c.pformat()
        self.assertEqual(formatted['rowcount'], 2)
        # Note: all dewrap and \ are to remove newlines and indentation cause by the triple quote.
        expected = textwrap.dedent("""\
        +-----+---------------+------+------------+
        | int |    bigint     | str  |   float    |
        +=====+===============+======+============+
        | 1   | 42            | str2 | 3.5        |
        | 1   | 1234567891234 | str1 | 4.45643245 |
        +-----+---------------+------+------------+""")
        self.assertEqual(formatted['table'], expected)

        # Sorted
        formatted_sorted = c.pformat(sort_by=[2])
        self.assertEqual(formatted['rowcount'], 2)
        # Note: all dewrap and \ are to remove newlines and indentation cause by the triple quote.
        expected_sorted = textwrap.dedent("""\
         +-----+---------------+------+------------+
         | int |    bigint     | str  |   float    |
         +=====+===============+======+============+
         | 1   | 1234567891234 | str1 | 4.45643245 |
         | 1   | 42            | str2 | 3.5        |
         +-----+---------------+------+------------+""")
        self.assertEqual(formatted_sorted['table'], expected_sorted)

    @patch('phos.Phos.description', new_callable=PropertyMock)
    @patch('pyhive.hive.Cursor.execute')
    def test_headers(self, super_exec, description):
        desc = desc = [
            ('int', 'int_type'),
            ('bigint', 'bigint_type'),
            ('str', 'string_type'),
            ('float', 'decimal_type')
        ]
        description.return_value = desc
        c = Phos(MagicMock())
        c.execute('exec needs to be called before fetchall', _cache=False)
        self.assertSequenceEqual([x[0] for x in desc], c.headers())

    @patch.object(pyhive.hive.Cursor, 'fetchall', return_value=[
        [1, 42, 'str1', 3.5],
        [1, 1234567891234, 'str2', 4.45643245]])
    @patch('phos.Phos.description', new_callable=PropertyMock)
    @patch('pyhive.hive.Cursor.execute')
    def test_df(self, super_exec, description, fetchall):
        """
        Overrides description and fetchall (part of Python DP API) to check output.
        """
        desc = [
            ('int', 'int_type'),
            ('bigint', 'bigint_type'),
            ('str', 'string_type'),
            ('float', 'decimal_type')
        ]
        description.return_value = desc
        c = Phos(MagicMock())
        c.execute('exec needs to be called before fetchall', _cache=False)
        df = c.get_df()
        self.assertSequenceEqual(c.headers(), list(df.columns), "Fetchall() and df do not have the same headers.")
        self.assertEqual(len(c.fetchall()), len(df), "Fetchall() and df do not have the same number of rows.")
        for r in df.itertuples(index=True):
            self.assertSequenceEqual(r[1:], c.fetchall()[r[0]], f"Rows {r[0]} of df and fetchall() differ.")

    @patch('seaborn.lineplot')
    @patch.object(pyhive.hive.Cursor, 'fetchall', return_value=[
        [1, 42, 'str1', 3.5],
        [1, 1234567891234, 'str2', 4.45643245]])
    @patch('phos.Phos.description', new_callable=PropertyMock)
    @patch('pyhive.hive.Cursor.execute')
    def test_plot_uses_proper_arguments(self, super_exec, description, fetchall, lineplot):
        """
        Overrides description and fetchall (part of Python DP API) to check output.
        """
        desc = [
            ('int', 'int_type'),
            ('bigint', 'bigint_type'),
            ('str', 'string_type'),
            ('float', 'decimal_type')
        ]
        description.return_value = desc
        c = Phos(MagicMock())
        c.execute('exec needs to be called before fetchall', _cache=False)
        c.plot_xyh(plot='line')
        lineplot.assert_called()
        with self.assertRaises(UnknownPlotTypeError):
            c.plot_xyh(plot='poofpoof')

# pprint
