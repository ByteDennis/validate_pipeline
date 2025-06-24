"""
PropertyMock - For Excel properties (font.size, alignment.horizontal)
MagicMock - For complex objects with multiple methods
Test file for xlwings format-related functionalities
Testing cell formatting, number formats, fonts, colors, borders, etc.
"""

import pytest
import xlwings as xw
from unittest.mock import Mock, MagicMock, patch, PropertyMock, call
import pandas as pd
from datetime import datetime


class TestXlwingsNumberFormat:
    """Test xlwings number formatting functionality"""
    
    @pytest.fixture
    def mock_range_with_format(self):
        """Mock Range with number_format property"""
        range_mock = MagicMock()
        # Use PropertyMock for number_format property
        type(range_mock).number_format = PropertyMock()
        return range_mock
    
    @pytest.mark.parametrize("format_code,expected", [
        ("General", "General"),
        ("0.00", "0.00"),
        ("#,##0.00", "#,##0.00"),
        ("$#,##0.00", "$#,##0.00"),
        ("0.00%", "0.00%"),
        ("m/d/yyyy", "m/d/yyyy"),
        ("h:mm:ss AM/PM", "h:mm:ss AM/PM"),
        ("@", "@"),  # Text format
    ])
    def test_number_format_codes(self, mock_range_with_format, format_code, expected):
        """Test various Excel number format codes"""
        # Set the format
        mock_range_with_format.number_format = format_code
        
        # Verify it was set correctly
        assert mock_range_with_format.number_format == expected
        
        # Verify the property was accessed
        type(mock_range_with_format).number_format.__set__.assert_called_with(
            mock_range_with_format, format_code
        )
    
    def test_number_format_reset(self, mock_range_with_format):
        """Test resetting number format to General"""
        mock_range_with_format.number_format = "General"
        assert mock_range_with_format.number_format == "General"
    
    def test_currency_format_locales(self, mock_range_with_format):
        """Test currency formats for different locales"""
        currency_formats = [
            "$#,##0.00",      # USD
            "€#,##0.00",      # EUR
            "£#,##0.00",      # GBP
            "¥#,##0",         # JPY
            "₹#,##0.00",      # INR
        ]
        
        for fmt in currency_formats:
            mock_range_with_format.number_format = fmt
            assert mock_range_with_format.number_format == fmt


class TestXlwingsFont:
    """Test xlwings font formatting functionality"""
    
    @pytest.fixture
    def mock_font(self):
        """Mock Font object with all properties"""
        font_mock = MagicMock()
        
        # Use PropertyMock for font properties
        type(font_mock).name = PropertyMock(return_value="Calibri")
        type(font_mock).size = PropertyMock(return_value=11.0)
        type(font_mock).bold = PropertyMock(return_value=False)
        type(font_mock).italic = PropertyMock(return_value=False)
        type(font_mock).underline = PropertyMock(return_value=False)
        type(font_mock).color = PropertyMock(return_value=(0, 0, 0))
        type(font_mock).strikethrough = PropertyMock(return_value=False)
        
        return font_mock
    
    @pytest.fixture
    def mock_range_with_font(self, mock_font):
        """Mock Range with font property"""
        range_mock = MagicMock()
        type(range_mock).font = PropertyMock(return_value=mock_font)
        return range_mock
    
    @pytest.mark.parametrize("font_name", [
        "Arial",
        "Times New Roman", 
        "Helvetica",
        "Courier New",
        "Comic Sans MS",
        "Tahoma"
    ])
    def test_font_name(self, mock_range_with_font, font_name):
        """Test setting different font names"""
        mock_range_with_font.font.name = font_name
        assert mock_range_with_font.font.name == font_name
    
    @pytest.mark.parametrize("font_size", [8, 10, 11, 12, 14, 16, 18, 24, 36, 48])
    def test_font_size(self, mock_range_with_font, font_size):
        """Test setting different font sizes"""
        mock_range_with_font.font.size = font_size
        assert mock_range_with_font.font.size == font_size
    
    @pytest.mark.parametrize("bold,italic,underline", [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ])
    def test_font_styles(self, mock_range_with_font, bold, italic, underline):
        """Test font style combinations"""
        mock_range_with_font.font.bold = bold
        mock_range_with_font.font.italic = italic
        mock_range_with_font.font.underline = underline
        
        assert mock_range_with_font.font.bold == bold
        assert mock_range_with_font.font.italic == italic
        assert mock_range_with_font.font.underline == underline
    
    @pytest.mark.parametrize("color", [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (128, 0, 128),    # Purple
        (0, 0, 0),        # Black
        (255, 255, 255),  # White
    ])
    def test_font_color_rgb(self, mock_range_with_font, color):
        """Test font colors using RGB values"""
        mock_range_with_font.font.color = color
        assert mock_range_with_font.font.color == color
    
    def test_font_strikethrough(self, mock_range_with_font):
        """Test font strikethrough property"""
        mock_range_with_font.font.strikethrough = True
        assert mock_range_with_font.font.strikethrough == True
        
        mock_range_with_font.font.strikethrough = False
        assert mock_range_with_font.font.strikethrough == False


class TestXlwingsAlignment:
    """Test xlwings alignment formatting functionality"""
    
    @pytest.fixture
    def mock_alignment(self):
        """Mock Alignment object"""
        alignment_mock = MagicMock()
        
        # Alignment properties
        type(alignment_mock).horizontal = PropertyMock(return_value="general")
        type(alignment_mock).vertical = PropertyMock(return_value="bottom")
        type(alignment_mock).wrap_text = PropertyMock(return_value=False)
        type(alignment_mock).shrink_to_fit = PropertyMock(return_value=False)
        type(alignment_mock).indent = PropertyMock(return_value=0)
        type(alignment_mock).text_rotation = PropertyMock(return_value=0)
        
        return alignment_mock
    
    @pytest.fixture
    def mock_range_with_alignment(self, mock_alignment):
        """Mock Range with alignment property"""
        range_mock = MagicMock()
        type(range_mock).alignment = PropertyMock(return_value=mock_alignment)
        return range_mock
    
    @pytest.mark.parametrize("horizontal", [
        "general", "left", "center", "right", "fill", "justify", 
        "center_across_selection", "distributed"
    ])
    def test_horizontal_alignment(self, mock_range_with_alignment, horizontal):
        """Test horizontal alignment options"""
        mock_range_with_alignment.alignment.horizontal = horizontal
        assert mock_range_with_alignment.alignment.horizontal == horizontal
    
    @pytest.mark.parametrize("vertical", [
        "top", "center", "bottom", "justify", "distributed"
    ])
    def test_vertical_alignment(self, mock_range_with_alignment, vertical):
        """Test vertical alignment options"""
        mock_range_with_alignment.alignment.vertical = vertical
        assert mock_range_with_alignment.alignment.vertical == vertical
    
    def test_wrap_text(self, mock_range_with_alignment):
        """Test wrap text functionality"""
        mock_range_with_alignment.alignment.wrap_text = True
        assert mock_range_with_alignment.alignment.wrap_text == True
        
        mock_range_with_alignment.alignment.wrap_text = False
        assert mock_range_with_alignment.alignment.wrap_text == False
    
    def test_shrink_to_fit(self, mock_range_with_alignment):
        """Test shrink to fit functionality"""
        mock_range_with_alignment.alignment.shrink_to_fit = True
        assert mock_range_with_alignment.alignment.shrink_to_fit == True
    
    @pytest.mark.parametrize("indent_level", [0, 1, 2, 3, 5, 10])
    def test_text_indent(self, mock_range_with_alignment, indent_level):
        """Test text indentation levels"""
        mock_range_with_alignment.alignment.indent = indent_level
        assert mock_range_with_alignment.alignment.indent == indent_level
    
    @pytest.mark.parametrize("rotation", [0, 45, 90, -45, -90, 180])
    def test_text_rotation(self, mock_range_with_alignment, rotation):
        """Test text rotation angles"""
        mock_range_with_alignment.alignment.text_rotation = rotation
        assert mock_range_with_alignment.alignment.text_rotation == rotation


class TestXlwingsBorders:
    """Test xlwings border formatting functionality"""
    
    @pytest.fixture
    def mock_border(self):
        """Mock Border object"""
        border_mock = MagicMock()
        
        # Border properties
        type(border_mock).line_style = PropertyMock(return_value="continuous")
        type(border_mock).weight = PropertyMock(return_value=2)
        type(border_mock).color = PropertyMock(return_value=(0, 0, 0))
        
        return border_mock
    
    @pytest.fixture
    def mock_borders(self, mock_border):
        """Mock Borders collection"""
        borders_mock = MagicMock()
        
        # Individual border sides
        borders_mock.left = mock_border
        borders_mock.right = mock_border
        borders_mock.top = mock_border
        borders_mock.bottom = mock_border
        borders_mock.inside_horizontal = mock_border
        borders_mock.inside_vertical = mock_border
        
        return borders_mock
    
    @pytest.fixture
    def mock_range_with_borders(self, mock_borders):
        """Mock Range with borders property"""
        range_mock = MagicMock()
        type(range_mock).borders = PropertyMock(return_value=mock_borders)
        return range_mock
    
    @pytest.mark.parametrize("line_style", [
        "continuous", "dash", "dash_dot", "dash_dot_dot", "dot", 
        "double", "none", "slant_dash_dot"
    ])
    def test_border_line_styles(self, mock_range_with_borders, line_style):
        """Test different border line styles"""
        mock_range_with_borders.borders.left.line_style = line_style
        assert mock_range_with_borders.borders.left.line_style == line_style
    
    @pytest.mark.parametrize("weight", [1, 2, 3, 4])
    def test_border_weights(self, mock_range_with_borders, weight):
        """Test border thickness/weight"""
        mock_range_with_borders.borders.top.weight = weight
        assert mock_range_with_borders.borders.top.weight == weight
    
    def test_individual_borders(self, mock_range_with_borders):
        """Test setting individual border sides"""
        # Set different styles for each side
        mock_range_with_borders.borders.left.line_style = "continuous"
        mock_range_with_borders.borders.right.line_style = "dash"
        mock_range_with_borders.borders.top.line_style = "dot"
        mock_range_with_borders.borders.bottom.line_style = "double"
        
        assert mock_range_with_borders.borders.left.line_style == "continuous"
        assert mock_range_with_borders.borders.right.line_style == "dash"
        assert mock_range_with_borders.borders.top.line_style == "dot"
        assert mock_range_with_borders.borders.bottom.line_style == "double"
    
    @pytest.mark.parametrize("border_color", [
        (255, 0, 0),    # Red
        (0, 128, 0),    # Green
        (0, 0, 255),    # Blue
        (128, 128, 128), # Gray
    ])
    def test_border_colors(self, mock_range_with_borders, border_color):
        """Test border colors"""
        mock_range_with_borders.borders.left.color = border_color
        assert mock_range_with_borders.borders.left.color == border_color


class TestXlwingsInterior:
    """Test xlwings cell interior/fill formatting"""
    
    @pytest.fixture
    def mock_interior(self):
        """Mock Interior object for cell background"""
        interior_mock = MagicMock()
        
        type(interior_mock).color = PropertyMock(return_value=(255, 255, 255))
        type(interior_mock).pattern = PropertyMock(return_value="solid")
        type(interior_mock).pattern_color = PropertyMock(return_value=(0, 0, 0))
        
        return interior_mock
    
    @pytest.fixture
    def mock_range_with_interior(self, mock_interior):
        """Mock Range with interior property"""
        range_mock = MagicMock()
        type(range_mock).interior = PropertyMock(return_value=mock_interior)
        return range_mock
    
    @pytest.mark.parametrize("fill_color", [
        (255, 255, 0),    # Yellow
        (0, 255, 255),    # Cyan
        (255, 0, 255),    # Magenta
        (192, 192, 192),  # Light Gray
        (255, 192, 203),  # Pink
    ])
    def test_cell_fill_colors(self, mock_range_with_interior, fill_color):
        """Test cell background fill colors"""
        mock_range_with_interior.interior.color = fill_color
        assert mock_range_with_interior.interior.color == fill_color
    
    @pytest.mark.parametrize("pattern", [
        "solid", "none", "gray75", "gray50", "gray25", "horizontal", 
        "vertical", "down", "up", "checker", "semi_gray75"
    ])
    def test_fill_patterns(self, mock_range_with_interior, pattern):
        """Test different fill patterns"""
        mock_range_with_interior.interior.pattern = pattern
        assert mock_range_with_interior.interior.pattern == pattern
    
    def test_pattern_color(self, mock_range_with_interior):
        """Test pattern color for patterned fills"""
        pattern_color = (128, 128, 128)
        mock_range_with_interior.interior.pattern_color = pattern_color
        assert mock_range_with_interior.interior.pattern_color == pattern_color


class TestXlwingsConditionalFormatting:
    """Test xlwings conditional formatting functionality"""
    
    @pytest.fixture
    def mock_format_condition(self):
        """Mock FormatCondition object"""
        condition_mock = MagicMock()
        
        type(condition_mock).type = PropertyMock(return_value=1)  # xlCellValue
        type(condition_mock).operator = PropertyMock(return_value=1)  # xlBetween
        type(condition_mock).formula1 = PropertyMock(return_value="0")
        type(condition_mock).formula2 = PropertyMock(return_value="100")
        
        # Format properties
        condition_mock.font = MagicMock()
        condition_mock.interior = MagicMock()
        condition_mock.borders = MagicMock()
        
        return condition_mock
    
    @pytest.fixture
    def mock_format_conditions(self, mock_format_condition):
        """Mock FormatConditions collection"""
        conditions_mock = MagicMock()
        conditions_mock.add = MagicMock(return_value=mock_format_condition)
        conditions_mock.delete = MagicMock()
        conditions_mock.__len__ = MagicMock(return_value=1)
        conditions_mock.__getitem__ = MagicMock(return_value=mock_format_condition)
        
        return conditions_mock
    
    @pytest.fixture
    def mock_range_with_conditional_format(self, mock_format_conditions):
        """Mock Range with conditional formatting"""
        range_mock = MagicMock()
        type(range_mock).format_conditions = PropertyMock(return_value=mock_format_conditions)
        return range_mock
    
    def test_add_conditional_format(self, mock_range_with_conditional_format):
        """Test adding conditional formatting rule"""
        # Add a condition
        condition = mock_range_with_conditional_format.format_conditions.add(
            type=1, operator=1, formula1="0", formula2="100"
        )
        
        # Verify the condition was added
        mock_range_with_conditional_format.format_conditions.add.assert_called_once_with(
            type=1, operator=1, formula1="0", formula2="100"
        )
        assert condition is not None
    
    @pytest.mark.parametrize("condition_type,operator,formula1,formula2", [
        (1, 1, "0", "100"),      # Between 0 and 100
        (1, 2, "50", None),      # Not between
        (1, 3, "75", None),      # Equal to 75
        (1, 4, "50", None),      # Not equal to 50
        (1, 5, "100", None),     # Greater than 100
        (1, 6, "0", None),       # Less than 0
    ])
    def test_conditional_format_operators(self, mock_range_with_conditional_format, 
                                        condition_type, operator, formula1, formula2):
        """Test different conditional formatting operators"""
        condition = mock_range_with_conditional_format.format_conditions.add(
            type=condition_type, operator=operator, formula1=formula1, formula2=formula2
        )
        
        mock_range_with_conditional_format.format_conditions.add.assert_called_with(
            type=condition_type, operator=operator, formula1=formula1, formula2=formula2
        )
    
    def test_conditional_format_styling(self, mock_range_with_conditional_format):
        """Test styling conditional format rules"""
        # Get a condition
        condition = mock_range_with_conditional_format.format_conditions[0]
        
        # Set font formatting
        condition.font.color = (255, 0, 0)  # Red font
        condition.font.bold = True
        
        # Set interior formatting
        condition.interior.color = (255, 255, 0)  # Yellow background
        
        # Verify formatting was applied
        assert condition.font.color == (255, 0, 0)
        assert condition.font.bold == True
        assert condition.interior.color == (255, 255, 0)
    
    def test_delete_conditional_format(self, mock_range_with_conditional_format):
        """Test deleting conditional formatting rules"""
        mock_range_with_conditional_format.format_conditions.delete()
        mock_range_with_conditional_format.format_conditions.delete.assert_called_once()


class TestXlwingsColumnWidth:
    """Test xlwings column width and row height formatting"""
    
    @pytest.fixture
    def mock_range_with_dimensions(self):
        """Mock Range with column and row dimension properties"""
        range_mock = MagicMock()
        
        # Column width properties
        type(range_mock).column_width = PropertyMock(return_value=8.43)
        type(range_mock).row_height = PropertyMock(return_value=15.0)
        
        # AutoFit methods
        range_mock.autofit = MagicMock()
        range_mock.columns.autofit = MagicMock()
        range_mock.rows.autofit = MagicMock()
        
        return range_mock
    
    @pytest.mark.parametrize("width", [5.0, 8.43, 12.0, 15.5, 20.0, 25.5])
    def test_column_width(self, mock_range_with_dimensions, width):
        """Test setting column widths"""
        mock_range_with_dimensions.column_width = width
        assert mock_range_with_dimensions.column_width == width
    
    @pytest.mark.parametrize("height", [12.75, 15.0, 18.0, 24.0, 30.0])
    def test_row_height(self, mock_range_with_dimensions, height):
        """Test setting row heights"""
        mock_range_with_dimensions.row_height = height
        assert mock_range_with_dimensions.row_height == height
    
    def test_autofit_columns(self, mock_range_with_dimensions):
        """Test auto-fitting column widths"""
        mock_range_with_dimensions.columns.autofit()
        mock_range_with_dimensions.columns.autofit.assert_called_once()
    
    def test_autofit_rows(self, mock_range_with_dimensions):
        """Test auto-fitting row heights"""
        mock_range_with_dimensions.rows.autofit()
        mock_range_with_dimensions.rows.autofit.assert_called_once()
    
    def test_autofit_range(self, mock_range_with_dimensions):
        """Test auto-fitting entire range"""
        mock_range_with_dimensions.autofit()
        mock_range_with_dimensions.autofit.assert_called_once()


class TestXlwingsFormattingIntegration:
    """Integration tests for combined formatting operations"""
    
    @pytest.fixture
    def mock_fully_formatted_range(self):
        """Mock Range with all formatting properties"""
        range_mock = MagicMock()
        
        # Number format
        type(range_mock).number_format = PropertyMock(return_value="General")
        
        # Font
        font_mock = MagicMock()
        type(font_mock).name = PropertyMock(return_value="Calibri")
        type(font_mock).size = PropertyMock(return_value=11.0)
        type(font_mock).bold = PropertyMock(return_value=False)
        type(font_mock).color = PropertyMock(return_value=(0, 0, 0))
        type(range_mock).font = PropertyMock(return_value=font_mock)
        
        # Alignment
        alignment_mock = MagicMock()
        type(alignment_mock).horizontal = PropertyMock(return_value="general")
        type(alignment_mock).vertical = PropertyMock(return_value="bottom")
        type(range_mock).alignment = PropertyMock(return_value=alignment_mock)
        
        # Interior
        interior_mock = MagicMock()
        type(interior_mock).color = PropertyMock(return_value=(255, 255, 255))
        type(range_mock).interior = PropertyMock(return_value=interior_mock)
        
        # Dimensions
        type(range_mock).column_width = PropertyMock(return_value=8.43)
        type(range_mock).row_height = PropertyMock(return_value=15.0)
        
        return range_mock
    
    def test_complete_cell_formatting(self, mock_fully_formatted_range):
        """Test applying complete cell formatting"""
        range_obj = mock_fully_formatted_range
        
        # Apply comprehensive formatting
        range_obj.number_format = "$#,##0.00"
        range_obj.font.name = "Arial"
        range_obj.font.size = 12
        range_obj.font.bold = True
        range_obj.font.color = (255, 0, 0)
        range_obj.alignment.horizontal = "center"
        range_obj.alignment.vertical = "center"
        range_obj.interior.color = (255, 255, 0)
        range_obj.column_width = 15.0
        range_obj.row_height = 20.0
        
        # Verify all formatting was applied
        assert range_obj.number_format == "$#,##0.00"
        assert range_obj.font.name == "Arial"
        assert range_obj.font.size == 12
        assert range_obj.font.bold == True
        assert range_obj.font.color == (255, 0, 0)
        assert range_obj.alignment.horizontal == "center"
        assert range_obj.alignment.vertical == "center"
        assert range_obj.interior.color == (255, 255, 0)
        assert range_obj.column_width == 15.0
        assert range_obj.row_height == 20.0
    
    def test_table_formatting_workflow(self, mock_fully_formatted_range):
        """Test typical table formatting workflow"""
        header_range = mock_fully_formatted_range
        
        # Header formatting
        header_range.font.bold = True
        header_range.font.color = (255, 255, 255)  # White text
        header_range.interior.color = (0, 0, 128)   # Navy background
        header_range.alignment.horizontal = "center"
        header_range.alignment.vertical = "center"
        
        # Data formatting  
        data_range = mock_fully_formatted_range
        data_range.number_format = "#,##0.00"
        data_range.font.name = "Calibri"
        data_range.font.size = 10
        data_range.alignment.horizontal = "right"
        
        # Verify header formatting
        assert header_range.font.bold == True
        assert header_range.font.color == (255, 255, 255)
        assert header_range.interior.color == (0, 0, 128)
        assert header_range.alignment.horizontal == "center"
        
        # Verify data formatting
        assert data_range.number_format == "#,##0.00"
        assert data_range.font.name == "Calibri"
        assert data_range.font.size == 10
        assert data_range.alignment.horizontal == "right"


class TestFormatCopyPaste:
    """Test format copying and pasting functionality"""
    
    @pytest.fixture
    def mock_range_with_copy_paste(self):
        """Mock Range with copy/paste format methods"""
        range_mock = MagicMock()
        range_mock.copy = MagicMock()
        range_mock.paste = MagicMock()
        range_mock.paste_special = MagicMock()
        return range_mock
    
    def test_copy_format(self, mock_range_with_copy_paste):
        """Test copying cell format"""
        source_range = mock_range_with_copy_paste
        source_range.copy()
        source_range.copy.assert_called_once()
    
    def test_paste_format_only(self, mock_range_with_copy_paste):
        """Test pasting format only (not values)"""
        target_range = mock_range_with_copy_paste
        target_range.paste_special(paste="formats")
        target_range.paste_special.assert_called_once_with(paste="formats")
    
    def test_paste_values_and_formats(self, mock_range_with_copy_paste):
        """Test pasting both values and formats"""
        target_range = mock_range_with_copy_paste
        target_range.paste_special(paste="all")
        target_range.paste_special.assert_called_once_with(paste="all")


if __name__ == "__main__":
    # Run the tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure for debugging
    ])