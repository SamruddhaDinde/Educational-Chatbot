import sys
class CustomeException(Exception):

    def __init__(self, message: str,error_detail:Exception = None):
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(message: str, error_detail):
        _, _, exception_traceback = sys.exc_info()
        file_name = exception_traceback.tb_frame.f_code.co_filename if exception_traceback else 'Unknown file'
        line_number = exception_traceback.tb_lineno if exception_traceback else 'Unknown line'
        return f"{message} | Error: {error_detail} | File: {file_name} | Line: {line_number}"
    
    def __str__(self):
        return self.error_message