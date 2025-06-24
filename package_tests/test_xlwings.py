
import pytest
import xlwings as xw
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pandas as pd
import numpy as np
from datetime import datetime, date
import os
import tempfile


class TestXlwingsApp:
    """Test xlwings App functionality"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock xlwings App"""
        with patch('xlwings.App') as mock_app_class:
            app_instance = Mock()
            mock_app_class.return_value = app_instance
            app_instance.visible = True
            app_instance.display_alerts = True
            app_instance.screen_updating = True
            app_instance.books = Mock()
            yield app_instance
    
    def test_app_creation_default(self, mock_app):
        """Test creating App with default settings"""
        app = xw.App()
        assert app is not None
        assert app.visible is True
    
    @pytest.mark.parametrize("visible,alerts,screen_update", [
        (True, True, True),
        (False, False, False),
        (True, False, True),
    ])
    def test_app_creation_with_params(self, mock_app, visible, alerts, screen_update):
        """Test creating App with various parameter combinations"""
        app = xw.App(visible=visible, add_book=False)
        app.display_alerts = alerts
        app.screen_updating = screen_update
        
        assert app.display_alerts == alerts
        assert app.screen_updating == screen_update
    
    def test_app_quit(self, mock_app):
        """Test App quit functionality"""
        app = xw.App()
        app.quit = Mock()
        app.quit()
        app.quit.assert_called_once()


class TestXlwingsBook:
    """Test xlwings Book functionality"""
    
    @pytest.fixture
    def mock_book(self):
        """Mock xlwings Book"""
        book = Mock()
        book.name = "TestBook.xlsx"
        book.fullname = "/path/to/TestBook.xlsx"
        book.app = Mock()
        book.sheets = Mock()
        book.save = Mock()
        book.close = Mock()
        return book
    
    @pytest.fixture
    def mock_books_collection(self, mock_book):
        """Mock Books collection"""
        books = Mock()
        books.open = Mock(return_value=mock_book)
        books.add = Mock(return_value=mock_book)
        books.__iter__ = Mock(return_value=iter([mock_book]))
        books.__len__ = Mock(return_value=1)
        return books
    
    def test_book_open(self, mock_books_collection, mock_book):
        """Test opening a book"""
        with patch('xlwings.books', mock_books_collection):
            book = xw.books.open("TestBook.xlsx")
            assert book.name == "TestBook.xlsx"
            mock_books_collection.open.assert_called_once_with("TestBook.xlsx")
    
    def test_book_add(self, mock_books_collection, mock_book):
        """Test adding a new book"""
        with patch('xlwings.books', mock_books_collection):
            book = xw.books.add()
            assert book is not None
            mock_books_collection.add.assert_called_once()
    
    def test_book_save(self, mock_book):
        """Test saving a book"""
        mock_book.save()
        mock_book.save.assert_called_once()
    
    @pytest.mark.parametrize("filename", [
        "test.xlsx",
        "/full/path/test.xlsx",
        "test_with_spaces.xlsx"
    ])
    def test_book_save_as(self, mock_book, filename):
        """Test saving book with different filenames"""
        mock_book.save_as = Mock()
        mock_book.save_as(filename)
        mock_book.save_as.assert_called_once_with(filename)
    
    def test_book_close(self, mock_book):
        """Test closing a book"""
        mock_book.close()
        mock_book.close.assert_called_once()


class TestXlwingsSheet:
    """Test xlwings Sheet functionality"""
    
    @pytest.fixture
    def mock_sheet(self):
        """Mock xlwings Sheet"""
        sheet = Mock()
        sheet.name = "Sheet1"
        sheet.book = Mock()
        sheet.range = Mock()
        sheet.clear = Mock()
        sheet.delete = Mock()
        return sheet
    
    @pytest.fixture
    def mock_sheets_collection(self, mock_sheet):
        """Mock Sheets collection"""
        sheets = Mock()
        sheets.add = Mock(return_value=mock_sheet)
        sheets.__getitem__ = Mock(return_value=mock_sheet)
        sheets.__iter__ = Mock(return_value=iter([mock_sheet]))
        sheets.active = mock_sheet
        return sheets
    
    def test_sheet_access_by_name(self, mock_sheets_collection, mock_sheet):
        """Test accessing sheet by name"""
        with patch('xlwings.sheets', mock_sheets_collection):
            sheet = xw.sheets["Sheet1"]
            assert sheet.name == "Sheet1"
            mock_sheets_collection.__getitem__.assert_called_once_with("Sheet1")
    
    def test_sheet_add(self, mock_sheets_collection, mock_sheet):
        """Test adding a new sheet"""
        with patch('xlwings.sheets', mock_sheets_collection):
            sheet = xw.sheets.add("NewSheet")
            assert sheet is not None
            mock_sheets_collection.add.assert_called_once_with("NewSheet")
    
    @pytest.mark.parametrize("sheet_name", [
        "Data",
        "Analysis_2023",
        "Sheet with spaces",
        "工作表1"  # Unicode test
    ])
    def test_sheet_name_variations(self, mock_sheets_collection, mock_sheet, sheet_name):
        """Test sheet names with various formats"""
        mock_sheet.name = sheet_name
        with patch('xlwings.sheets', mock_sheets_collection):
            sheet = xw.sheets[sheet_name]
            assert sheet.name == sheet_name
    
    def test_sheet_clear(self, mock_sheet):
        """Test clearing sheet contents"""
        mock_sheet.clear()
        mock_sheet.clear.assert_called_once()
    
    def test_sheet_delete(self, mock_sheet):
        """Test deleting a sheet"""
        mock_sheet.delete()
        mock_sheet.delete.assert_called_once()


class TestXlwingsRange:
    """Test xlwings Range functionality"""
    
    @pytest.fixture
    def mock_range(self):
        """Mock xlwings Range"""
        range_mock = Mock()
        range_mock.value = None
        range_mock.formula = None
        range_mock.address = "A1"
        range_mock.sheet = Mock()
        range_mock.clear = Mock()
        range_mock.expand = Mock(return_value=range_mock)
        return range_mock
    
    @pytest.fixture
    def mock_sheet_with_range(self, mock_range):
        """Mock Sheet that returns Range"""
        sheet = Mock()
        sheet.range = Mock(return_value=mock_range)
        return sheet
    
    @pytest.mark.parametrize("address", [
        "A1",
        "B2:D4", 
        "Sheet1!A1:C3",
        "A:A",  # Entire column
        "1:1"   # Entire row
    ])
    def test_range_address_formats(self, mock_sheet_with_range, mock_range, address):
        """Test various range address formats"""
        mock_range.address = address
        range_obj = mock_sheet_with_range.range(address)
        assert range_obj.address == address
        mock_sheet_with_range.range.assert_called_once_with(address)
    
    @pytest.mark.parametrize("value,expected", [
        (42, 42),
        ("Hello", "Hello"),
        (3.14159, 3.14159),
        (True, True),
        (None, None),
        (datetime(2023, 12, 25), datetime(2023, 12, 25))
    ])
    def test_range_value_types(self, mock_range, value, expected):
        """Test setting and getting various value types"""
        mock_range.value = value
        assert mock_range.value == expected
    
    def test_range_value_list(self, mock_range):
        """Test range with list values (arrays)"""
        test_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        mock_range.value = test_data
        assert mock_range.value == test_data
    
    @pytest.mark.parametrize("formula", [
        "=SUM(A1:A10)",
        "=VLOOKUP(A1,B:C,2,FALSE)",
        "=IF(A1>0,A1*2,0)",
        "=TODAY()",
        "=CONCATENATE(A1,\" \",B1)"
    ])
    def test_range_formulas(self, mock_range, formula):
        """Test setting various Excel formulas"""
        mock_range.formula = formula
        assert mock_range.formula == formula
    
    def test_range_clear(self, mock_range):
        """Test clearing range contents"""
        mock_range.clear()
        mock_range.clear.assert_called_once()
    
    def test_range_expand(self, mock_range):
        """Test range expansion"""
        expanded = mock_range.expand()
        assert expanded is not None
        mock_range.expand.assert_called_once()


class TestXlwingsDataFrameIntegration:
    """Test xlwings DataFrame integration"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing"""
        return pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Salary': [50000.0, 60000.0, 70000.0],
            'Date': [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)]
        })
    
    @pytest.fixture
    def mock_range_with_dataframe(self):
        """Mock Range that handles DataFrame operations"""
        range_mock = Mock()
        range_mock.options = Mock(return_value=range_mock)
        range_mock.value = None
        return range_mock
    
    def test_dataframe_to_excel(self, mock_range_with_dataframe, sample_dataframe):
        """Test writing DataFrame to Excel"""
        # Mock the options chain
        options_mock = Mock()
        options_mock.value = sample_dataframe
        mock_range_with_dataframe.options.return_value = options_mock
        
        # Simulate writing DataFrame
        mock_range_with_dataframe.options(pd.DataFrame).value = sample_dataframe
        
        # Verify the chain was called
        mock_range_with_dataframe.options.assert_called_once()
    
    def test_dataframe_from_excel(self, mock_range_with_dataframe, sample_dataframe):
        """Test reading DataFrame from Excel"""
        # Mock return value as DataFrame
        mock_range_with_dataframe.options.return_value.value = sample_dataframe
        
        # Simulate reading DataFrame
        result = mock_range_with_dataframe.options(pd.DataFrame, header=1, index=False).value
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        mock_range_with_dataframe.options.assert_called_once()
    
    @pytest.mark.parametrize("header,index", [
        (True, False),
        (False, True), 
        (True, True),
        (False, False)
    ])
    def test_dataframe_options(self, mock_range_with_dataframe, sample_dataframe, header, index):
        """Test DataFrame options combinations"""
        options_mock = Mock()
        options_mock.value = sample_dataframe
        mock_range_with_dataframe.options.return_value = options_mock
        
        # Test different option combinations
        mock_range_with_dataframe.options(pd.DataFrame, header=header, index=index).value = sample_dataframe
        
        mock_range_with_dataframe.options.assert_called_with(pd.DataFrame, header=header, index=index)


class TestXlwingsErrorHandling:
    """Test xlwings error handling"""
    
    @pytest.fixture
    def mock_failing_operations(self):
        """Mock operations that can fail"""
        mock_obj = Mock()
        mock_obj.failing_operation = Mock(side_effect=Exception("Test error"))
        return mock_obj
    
    def test_com_error_handling(self, mock_failing_operations):
        """Test handling COM errors"""
        with pytest.raises(Exception) as exc_info:
            mock_failing_operations.failing_operation()
        
        assert "Test error" in str(exc_info.value)
    
    def test_file_not_found_error(self):
        """Test file not found error handling"""
        with patch('xlwings.books') as mock_books:
            mock_books.open.side_effect = FileNotFoundError("File not found")
            
            with pytest.raises(FileNotFoundError):
                xw.books.open("nonexistent.xlsx")
    
    def test_invalid_range_error(self):
        """Test invalid range address error"""
        with patch('xlwings.sheets') as mock_sheets:
            mock_sheet = Mock()
            mock_sheet.range.side_effect = ValueError("Invalid range")
            mock_sheets.__getitem__.return_value = mock_sheet
            
            with pytest.raises(ValueError):
                xw.sheets["Sheet1"].range("INVALID_RANGE")


class TestXlwingsUtilities:
    """Test xlwings utility functions"""
    
    def test_view_function(self):
        """Test view function with mock"""
        test_data = [[1, 2, 3], [4, 5, 6]]
        
        with patch('xlwings.view') as mock_view:
            xw.view(test_data)
            mock_view.assert_called_once_with(test_data)
    
    @pytest.mark.parametrize("data_type,test_data", [
        ("list", [1, 2, 3, 4, 5]),
        ("dict", {"a": 1, "b": 2, "c": 3}),
        ("numpy_array", np.array([1, 2, 3, 4, 5])),
        ("pandas_series", pd.Series([1, 2, 3, 4, 5]))
    ])
    def test_view_different_data_types(self, data_type, test_data):
        """Test view function with different data types"""
        with patch('xlwings.view') as mock_view:
            xw.view(test_data)
            mock_view.assert_called_once_with(test_data)

class TestXlwingsIntegrationScenarios:
    """Integration test scenarios"""
    
    @pytest.fixture
    def mock_excel_workflow(self):
        """Mock complete Excel workflow"""
        # Mock App
        app = Mock()
        app.visible = True
        
        # Mock Book
        book = Mock()
        book.name = "TestWorkbook.xlsx"
        app.books.open.return_value = book
        
        # Mock Sheet
        sheet = Mock()
        sheet.name = "Sheet1"
        book.sheets.__getitem__.return_value = sheet
        
        # Mock Range
        range_mock = Mock()
        range_mock.value = None
        sheet.range.return_value = range_mock
        
        return {
            'app': app,
            'book': book, 
            'sheet': sheet,
            'range': range_mock
        }
    
    def test_complete_workflow(self, mock_excel_workflow):
        """Test complete xlwings workflow"""
        workflow = mock_excel_workflow
        
        # Simulate opening workbook
        book = workflow['app'].books.open("TestWorkbook.xlsx")
        assert book.name == "TestWorkbook.xlsx"
        
        # Simulate accessing sheet
        sheet = book.sheets["Sheet1"]
        assert sheet.name == "Sheet1"
        
        # Simulate writing data
        test_data = [[1, 2, 3], [4, 5, 6]]
        range_obj = sheet.range("A1")
        range_obj.value = test_data
        
        # Verify operations
        workflow['app'].books.open.assert_called_once_with("TestWorkbook.xlsx")
        book.sheets.__getitem__.assert_called_once_with("Sheet1")
        sheet.range.assert_called_once_with("A1")
        assert range_obj.value == test_data
    
    def test_dataframe_roundtrip(self, mock_excel_workflow):
        """Test DataFrame write and read roundtrip"""
        workflow = mock_excel_workflow
        
        # Original DataFrame
        original_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [1.1, 2.2, 3.3]
        })
        
        # Mock writing DataFrame
        range_obj = workflow['range']
        options_mock = Mock()
        options_mock.value = original_df
        range_obj.options.return_value = options_mock
        
        # Write DataFrame
        range_obj.options(pd.DataFrame).value = original_df
        
        # Read DataFrame back
        read_df = range_obj.options(pd.DataFrame, header=1).value
        
        # Verify
        assert isinstance(read_df, pd.DataFrame)
        range_obj.options.assert_called()


# # Conftest.py content for shared fixtures
# @pytest.fixture(scope="session")
# def excel_test_file():
#     """Create temporary Excel file for testing"""
#     with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
#         # Create a simple Excel file using pandas
#         df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
#         df.to_excel(tmp.name, index=False)
#         yield tmp.name
    
#     # Cleanup
#     if os.path.exists(tmp.name):
#         os.unlink(tmp.name)
