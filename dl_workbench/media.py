import base64
import binascii
import csv
import io
import json
from zipfile import ZIP_DEFLATED, ZipFile


class InputError(ValueError):
    pass


def binary_input(value, description):
    if not isinstance(value, str) or not value:
        raise InputError(f"请选择{description}或加载示例。")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise InputError(f"{description}数据格式有误。") from error


def encode(data):
    return base64.b64encode(data).decode("ascii")


def table_csv(columns, rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(columns)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def dataset_zip(files, annotations):
    output = io.BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, data in files:
            archive.writestr(name, data)
        archive.writestr("annotations.json", json.dumps(annotations, ensure_ascii=False, indent=2).encode("utf-8"))
    return encode(output.getvalue())
