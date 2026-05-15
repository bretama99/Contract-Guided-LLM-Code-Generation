from openpyxl.styles import Font, Alignment, Side, Border
from openpyxl.utils import get_column_letter

def format_worksheet(worksheet):
    
    worksheet.sheet_view.showGridLines = True
    worksheet.freeze_panes = "A2"
    worksheet.sheet_view.zoomScale = 75

    # fonts
    header_font = Font(name="Times New Roman", size=14, bold=True)
    body_font = Font(name="Aptos", size=12)

    # separator line style
    thin_gray = Side(style="thin", color="D9D9D9")
    border = Border(
        left=thin_gray,
        right=thin_gray,
        top=thin_gray,
        bottom=thin_gray,
    )

    max_row = worksheet.max_row
    max_col = worksheet.max_column

    # format all cells
    for row in worksheet.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
        for cell in row:
            cell.font = body_font
            cell.border = border
            cell.alignment = Alignment(
                horizontal="left",
                vertical="center",
                wrap_text=False,
            )

    # format header
    for cell in worksheet[1]:
        cell.font = header_font
        cell.border = border
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=False,
        )
        
    # size columns
    for column_cells in worksheet.columns:
        max_length = 0
        column_letter = column_cells[0].column_letter

        for cell in column_cells:
            if cell.value is not None:
                cell_length = len(str(cell.value))
                max_length = max(max_length, cell_length)

        adjusted_width = min(max(max_length * 1.2, 15), 70)
        worksheet.column_dimensions[column_letter].width = adjusted_width

    # wrap text for all cells
    for row in worksheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True
            )