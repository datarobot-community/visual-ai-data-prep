#!/usr/bin/env python

"""
        python visualai_data_prep.py <input.csv> <output.csv> <image_col>

        parameters:
            input           : string, required
                              input csv file, using delim = ',' and quote = '"',
                              data should be UTF-8 or printable ASCII to be safe
                              records line delimitered with multiline text support when using quoted strings
            output          : string, required
                              iff the output extension is '.csv' then the image data is encoded as base64 and
                              written to the csv file into the image-dst-col.
                              in all other situations the output will be treated as a directory and images
                              will be saved into this folder and the resulting data csv will be named master.csv
                              in both cases the resulting output csv file will be written using
                              record delimitered = newline,  field delim = ',' and with the quote = '"'
            image_src_col   : string, required
                              column name containing either the url/path/base64 image information to use

        options:
            --image_dst_col : string, default = image-src-col
                              column name to create or use in the output csv file for the image file/base64 data
                              if the provided and differs to image_col
            --resize        : string, default = 224x224 (this is what models use)
                              in the form of WIDTHxHEIGHT of the normalised image in the output
            --keep_aspect   : boolean, default = False
                              True, the image aspect is unchanged and is only resized to fit within the resize value
                              Fasle, the image is resized to exactly the resize value
            --threads       : integer, default = half of the SystemValue, range = 1-SystemValue
                              to enable faster image processing, multiple records can be done in parallel.
            --zip           : flag
                              after completing conversion, zip the output file/folder
            --debug         : flag
                              set to enable debug level logging
            --help
"""


import sys, os, base64, hashlib, logging, argparse
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
import requests
from queue import Queue, Empty
from threading import Thread
from threading import Lock
import csv, json, zipfile
from multiprocessing.dummy import Pool as ThreadPool


# global variables
VERSION = 'Version 2.1 2020-11-30'
IMAGENET = 224
SCALE = 1
RESIZE=(int(IMAGENET*SCALE),int(IMAGENET*SCALE))
try:
    CPU_COUNT = os.cpu_count()
except:
    CPU_COUNT = 4

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    #format="%(asctime)s %(levelname)s %(message)s",
    format='%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)
log.info(VERSION)

# boto3 client creation is not thread safe so need to create a lock to manage it
boto3_client_lock = Lock()
pwd = os.getcwd()

def get_s3_bytes(parsed_uri):
    log.debug('get_s3_bytes')
    import boto3
    try:
        bucketname = parsed_uri.netloc
        object_key = parsed_uri.path[1:]
        log.info(f'get s3  : {bucketname}, {object_key}')
        with boto3_client_lock:
            s3 = boto3.resource('s3')
            obj = s3.Object(bucketname, object_key)
        buffer = BytesIO()
        buffer.write(obj.get()['Body'].read())
    except Exception as e:
        log.error(e)
        buffer = None
    return buffer

def get_url_bytes(parsed_uri):
    log.debug('get_url_bytes')
    try:
        url = parsed_uri.geturl()
        log.info(f'get url : {url}')
        buffer = BytesIO()
        buffer.write(requests.get(url).content)
        buffer.seek(0)
    except Exception as e:
        log.error(e)
        buffer = None
    return buffer

def get_localfile_bytes(parsed_uri):
    log.debug('get_localfile_bytes')
    try:
        filepath = parsed_uri.geturl()
        # the decision to chdir to the output dir is not a good one, but this
        # work around should do instead of refactoring completly
        if not os.path.isfile(filepath):
            origional_filepath = filepath
            filepath = os.path.join(pwd, filepath)
            log.debug(f'{origional_filepath} not found, trying {filepath} instead')
        log.info(f'get file: {filepath}')
        buffer = open(filepath, 'rb')
    except Exception as e:
        log.error(e)
        buffer = None
    return buffer

def base64_to_bytes(data):
    try:
        return BytesIO(base64.b64decode(data))
    except Exception as e:
        log.error(e)
        return None

def get_some_bytes(parsed_uri):
    if len(parsed_uri.geturl()) > 1024:
        # assuming this is not a filename  based on length
        buffer = base64_to_bytes(parsed_uri.geturl())
    else:
        buffer = get_localfile_bytes(parsed_uri) or base64_to_bytes(parsed_uri.path)
    return buffer

def get_unknown_bytes(parsed_uri):
    valid_schemes = [ key for key in get_bytes_switcher ]
    log.warning(f"Unknown scheme '{parsed_uri.scheme}', valid schemes are {valid_schemes}")
    return None

get_bytes_switcher = {
    's3': get_s3_bytes,
    'http': get_url_bytes,
    'https': get_url_bytes,
    'file': get_localfile_bytes,
    '' : get_some_bytes
}

def get_image_from_bytes(bytes):
    try:
        if bytes:
            image = Image.open(bytes)
            return image
    except Exception as e:
        log.error(e)
    return None

def get_bytes_from_image(image):
    try:
        if image:
            bytes = BytesIO()
            image.save(bytes, image.format)
            bytes.seek(0)
            return bytes
    except Exception as e:
        log.error(e)
    return None

def base64_to_bytes(data):
    try:
        return BytesIO(base64.b64decode(data))
    except Exception as e:
        log.error(e)
        return None

def bytes_to_base64(bytes):
    try:
        if bytes:
            # just be sure we're at the beginning of the stream
            bytes.seek(0)
            return base64.b64encode(bytes.read())
    except Exception as e:
        log.error(e)
    return None

def resize_image(image, size=None, force=True):
    if image and size:
        if force:
            new_image = image.resize(size)
            new_image.format = image.format
            image = new_image
        else:
            image.thumbnail(size)
    return image

def normalize_image(image_path, resize, b64_string=True, force_size=True):
    # if b64_string then return result as a base64 string
    # otherwise return local filename where saved
    parsed_uri = urlparse(image_path)
    # ParseResult(scheme='', netloc='', path='', params='', query='', fragment='')
    #
    # the idea initially looked ok to use urlparse to identify the method seemed
    # ok but it turns out to be not so great.  this could be refactored but it
    # is working at the moment.  The edge cases when it's a local file or when
    # it's a base64 utf-8 string image and when the data includes '/' urlparse sometimes
    # gets confused.  I've got work arounds in place which seem to be holding up ok.
    get_image_bytes = get_bytes_switcher.get(parsed_uri.scheme, get_unknown_bytes)
    # get_image_bytes is the get data function dependant on the parsed_uri.scheme
    #
    image_bytes = get_image_bytes(parsed_uri)
    # image_bytes is a _io.BufferedReader or a _io.BytesIO
    #
    origional_image = get_image_from_bytes(image_bytes)
    # origional_image is a PIL Image object
    #
    # build the new image information and transform the origional image
    new_image_name = hashlib.md5(image_path.encode('utf-8')).hexdigest()
    new_image_type = origional_image.format if origional_image else 'error'
    new_filename = f'{new_image_name}.{new_image_type.lower()}'
    new_image = resize_image(image=origional_image, size=resize, force=force_size)
    new_image_bytes = get_bytes_from_image(new_image)
    # new_image_bytes is a _io.BufferedReader
    #
    # the return value is either the encoded image data or the image filename
    if b64_string:
        # return the image encoded data
        result = bytes_to_base64(new_image_bytes).decode("utf-8")
        log.debug(f'put b64 : ...{result[-100:]}')
        return result
    else:
        # save the image file and return the image filename
        with open(new_filename, 'wb') as outfile:
            if new_image_bytes:
                log.debug(f"put file: {new_filename}")
                outfile.write(new_image_bytes.read())
        return new_filename
#
# create out record processor so that we can use multithreading
def process_row(row, options, writer):
    log.debug(f'row : {json.dumps(row)}')
    image_str = ''
    try:
        if options.image_src_col in row:
            image_str = normalize_image(
                row[options.image_src_col],
                resize = options.resize,
                b64_string = options.image_output_type == 'base64',
                force_size = not options.keep_aspect
            )
        else:
            log.error(f'error: {options.image_src_col} not in columns')
    except Exception as e:
        log.error(f'error: normalize_image failed, will put an empty image')
        log.exception(e)
    finally:
        row[options.image_dst_col] = image_str
        writer.writerow(row)
#
# to be safe with out parallel output file writes to the csv we're using
# a queue. queues are thread safe but and ordinaty file is not for write
class safewriter:
    def __init__(self, *args):
        self.filewriter = open(*args)
        self.queue = Queue()
        self.finished = False
        Thread(name = "safewriter", target=self.internal_writer).start()

    def write(self, data):
        self.queue.put(data)

    def internal_writer(self):
        while not self.finished:
            try:
                data = self.queue.get(True, 1)
            except Empty:
                continue
            self.filewriter.write(data)
            self.queue.task_done()

    def close(self):
        self.queue.join()
        self.finished = True
        self.filewriter.close()

def zip_result(zip_this):
    log.info(f'post processing, zipping {zip_this}')
    # if the payload is a file then just zip this file
    if  os.path.isfile(zip_this):
        filename, file_extension = os.path.splitext(zip_this)
        zip_filename = filename + '.zip'
        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            log.info(f'{zip_filename}: adding {zip_this}')
            zip_file.write(zip_this, compress_type=zipfile.ZIP_DEFLATED)
            os.remove(zip_this)
    # if the payload is a directory then add each file within (not walking)
    elif os.path.isdir(zip_this):
        zip_filename = zip_this + '.zip'
        files = [ file for file in os.listdir(zip_this) ]
        files.sort()
        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            pwd = os.getcwd()
            os.chdir(zip_this)
            for file in files:
                if os.path.isfile(file):
                    log.info(f'{zip_filename}: adding {file}')
                    zip_file.write(file, compress_type=zipfile.ZIP_DEFLATED)
                    os.remove(file)
                else:
                    log.error(f'not expecting nested folders in {zip_this}')
            os.chdir(pwd)
            os.rmdir(zip_this)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2size(v):
    if isinstance(v, str):
        w = h = 224
        wh = v.split('x')
        if len(wh) == 1:
            w = h = int(wh[0])
        elif len(wh) == 2:
            w = int(wh[0])
            h = int(wh[1])
        return (w, h)
    else:
        raise argparse.ArgumentTypeError('string value of format WIDTHxHEIGHT is expected eg (600x600)')

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, usage=f'''
        python %(prog)s <input.csv> <output.csv> <image_col>

        parameters:
            input           : string, required
                              input csv file, using delim = ',' and quote = '"',
                              data should be UTF-8 or printable ASCII to be safe
                              records line delimitered with multiline text support when using quoted strings
            output          : string, required
                              iff the output extension is '.csv' then the image data is encoded as base64 and
                              written to the csv file into the image-dst-col.
                              in all other situations the output will be treated as a directory and images
                              will be saved into this folder and the resulting data csv will be named master.csv
                              in both cases the resulting output csv file will be written using
                              record delimitered = newline,  field delim = ',' and with the quote = '"'
            image_src_col   : string, required
                              column name containing either the url/path/base64 image information to use

        options:
            --image_dst_col : string, default = image-src-col
                              column name to create or use in the output csv file for the image file/base64 data
                              if the provided and differs to image_col
            --resize        : string, default = {RESIZE[0]}x{RESIZE[1]}
                              in the form of WIDTHxHEIGHT of the normalised image in the output
            --keep_aspect   : boolean, default = False
                              True, the image aspect is unchanged and is only resized to fit within the resize value
                              Fasle, the image is resized to exactly the resize value
            --threads       : integer, default = {int(CPU_COUNT/2)}, range = 1-{CPU_COUNT}
                              to enable faster image processing, multiple records can be done in parallel.
            --zip           : flag
                              after completing conversion, zip the output file/folder
            --debug         : flag
                              set to enable debug level logging
        '''
    )
    parser.add_argument(
        'input',
        help='Input CSV file with data prepared'
    )
    parser.add_argument(
        'output',
        help='Output file/dir'
    )
    parser.add_argument(
        'image_src_col',
        help='column to source image information from'
    )
    parser.add_argument(
        '--image_dst_col',
        help='column to source image information from'
    )
    parser.add_argument(
        '--resize',
        type=str2size,
        default=RESIZE,
        help='optional: <width>x<height>, eg 600x600'
    )
    parser.add_argument(
        '--keep_aspect',
        type=str2bool,
        default=False,
        help='optional: true/[false]'
    )
    parser.add_argument(
        '--threads',
        type=int,
        choices=range(1,CPU_COUNT+1),
        default=int(CPU_COUNT/2 + 1),
        help='optional: process images in parallel'
    )
    parser.add_argument(
        '--debug',
        action='store_const',
        const=logging.DEBUG,
        default=logging.INFO,
        help='optional: enable debug level logging'
    )
    parser.add_argument(
        '--zip',
        action='store_const',
        const=True,
        default=False,
        help='optional: post preperation, zip file or directory'
    )

    result = parser.parse_args()

    log.setLevel(result.debug)

    # Post processing adjustments
    if result.image_dst_col == None:
        result.image_dst_col = result.image_src_col

    outfile_name, outfile_extension = os.path.splitext(result.output)
    if outfile_extension.lower() == '.csv':
        result.image_output_type = 'base64'
        result.output_type = 'file'
        result.output_csv_file = result.output
    else:
        result.image_output_type = 'file'
        result.output_type = 'directory'
        result.output_csv_file = 'master.csv'

    log.debug(result)

    return result

def main():
    #
    # process out command line arguments and get/generate our options for execution
    #
    options = parse_args()
    log.info(f'source : column {options.image_src_col} '
            f'in file {options.input}')
    log.info(f'result : column {options.image_dst_col} '
            f'of type {options.image_output_type} '
            f'in {options.output_type} '
            f'{options.output} with size '
            f'{"maximum" if options.keep_aspect else "exactly"} '
            f'{options.resize[0]}x{options.resize[0]} ')
    #
    # test input as file and set input_csv_file in options
    #
    try:
        if os.path.isfile(options.input):
            options.input_csv_file = os.path.abspath(options.input)
        else:
            log.error(f'{options.input} is not a file or does not exist')
            raise
    except Exception as e:
        log.error(f'failed to locate input file : {e}')
    #
    # if the output_type is directory,
    #   we need to create it
    #   and becase the other stuff I've done assumes current dir,
    #   lets keep it simple and just move there (I hope this is not a bad decision)
    #
    if options.output_type == 'directory':
        pwd = os.getcwd()
        try:
            os.mkdir(options.output)
        except OSError as e:
            log.error(f'filed to create output directory {options.output} : {e}')
        else:
            log.info("Successfully created directory %s " % options.output)
        os.chdir(options.output)
    #
    # open input_csv_file with csv reader
    fd_in = open(options.input_csv_file, 'r', encoding='utf-8')
    reader = csv.DictReader(fd_in)
    input_columns = reader.fieldnames
    #
    # place image_dst_col in the destination column set if required
    if options.image_dst_col in input_columns:
        output_columns = input_columns
    else:
        output_columns = input_columns + [options.image_dst_col]
    #
    # open out result csv file
    fd_out = open(options.output_csv_file, 'w')
    writer = csv.DictWriter(fd_out, output_columns)
    writer.writeheader()
    #
    # now lets process the images
    pool = ThreadPool(options.threads)
    results = pool.map(lambda r: process_row(r, options, writer), reader)
    pool.close()
    pool.join()
    #
    # critical for consistant output that we close the output safewriter
    fd_out.close()
    fd_in.close()
    #
    # return to our starting directory after building output set of type directory
    if options.output_type == 'directory':
        os.chdir(pwd)

    if options.zip:
        zip_result(options.output)

if __name__ == "__main__":
    main()
