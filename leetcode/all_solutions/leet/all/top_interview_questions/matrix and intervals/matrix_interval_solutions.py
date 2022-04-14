from typing import List


class MatrixAndIntervalSolutions:

    def isValidSudoku(self, board: List[List[str]]) -> bool:

        rows, col, boxes = set(), set(), set()

        for i in range(9):
            for j in range(9):

                if board[i][j] != ".":  # empty boxes are not valid
                    row_key = (i, board[i][j])
                    col_key = (i, board[i][j])
                    boxes_key = (i // 3, j // 3, board[i][j])

                    if row_key in rows or col_key in col or boxes_key in boxes:
                        return False  # duplicates not allowed

                    rows.add(row_key)
                    col.add(col_key)
                    boxes.add(boxes_key)

        return True
