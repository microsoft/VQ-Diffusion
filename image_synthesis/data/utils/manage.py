from sys import stdout
import zipfile
import os.path as osp
import lmdb
import logging
from PIL import Image
import pickle
import io
import glob
import os
from pathlib import Path
import time
from threading import Thread
from queue import Queue,Empty
import subprocess

def func_wrapper(func):
    def sub_func(queue,kwargs):
        while True:
            try:
                key=queue.get(False)
                ret=func(key,**kwargs)
            except Empty:
                break
    return sub_func

class ThreadPool:
    def __init__(self,n):
        self.threads=[]
        self.n=n

    def run(self,func,array,**kwargs):
        queue=Queue()
        for val in array:
            queue.put(val)
        threads=[]
        target=func_wrapper(func)
        # hold_thread=subprocess.Popen("exec "+"python /mnt/blob/datasets/holder.py",shell=True,stdout=subprocess.DEVNULL)
        time.sleep(1)
        print(f"start loading queue {queue.qsize()}")
        logging.info(f"start loading queue {queue.qsize()}")
        for i in range(self.n):            
            print(i)
            thread=Thread(target=target, args=(queue,kwargs))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        # hold_thread.kill()


home = str(Path.home())
abs_blob_path=os.path.realpath("/mnt/blob/")
CACHE_FOLDER=os.path.join(home,"caching")
USE_CACHE=True

def norm(path):
    assert "*" not in path
    return os.path.realpath(os.path.abspath(path))

def in_blob(file):
    if abs_blob_path in file:
        return True
    else:
        return False

def map_name(file):
    path=norm(file)
    path=path.lstrip(abs_blob_path+"/")
    path=path.replace("/","_")
    assert len(path)<250
    return path


def preload(db,sync=True,load=True):
    if not load:
        return
    print(f"loading {db.db_path}")
    logging.info(f"loading {db.db_path}")
    if sync:
        db.initialize()
    else:
        p = Thread(target=db.initialize)
        p.start()

def get_keys_from_lmdb(db):
    with db.begin(write=False) as txn:
        return list(txn.cursor().iternext(values=False))

def decode_img(byteflow):
    img=Image.open(io.BytesIO(byteflow)).convert("RGB")
    img.load()
    return img

def decode_text(byteflow):
    return pickle.loads(byteflow)
    
decode_funcs={
    "image": decode_img,
    "text": decode_text
}

class MultipleZipManager:
    def __init__(self, files: list):
        raise

def remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text

class ZipManager:
    def __init__(self, db_path,data_type,prefix=None,load=True) -> None:
        self.decode_func=decode_funcs[data_type]

        self.db_path=db_path

        cache_file=os.path.join(CACHE_FOLDER,map_name(db_path))

        if USE_CACHE and os.path.exists(cache_file):
            logging.info(f"using local cache {cache_file}")
            self.db_path=cache_file

        if prefix is None:
            self.prefix = None
        else:
            self.prefix=f"{prefix}_"
        
        self._init=False
        preload(self,load=load)
        
    def deinitialze(self):
        self.zip_fd.close()
        del self.zip_fd
        self._init = False

    def initialize(self,close=True):
        self.zip_fd = zipfile.ZipFile(self.db_path, mode="r")
        if not hasattr(self,"_keys"):
            self._keys = self.zip_fd.namelist()
            if self.prefix is not None:
                self._keys=[self.prefix+key for key in self._keys]
        self._init = True
        if close:
            self.deinitialze()
        
    @property
    def keys(self):
        while not hasattr(self,"_keys"):
            time.sleep(0.1)
        return self._keys

    def get(self, name):
        if not self._init:
            self.initialize(close=False)  # https://discuss.pytorch.org/t/dataloader-stucks/14087/3
        byteflow = self.zip_fd.read(name)
        return self.decode_func(byteflow)

class DBManager:
    def __init__(self, db_path,data_type,prefix=None,load=True) -> None:
        self.decode_func=decode_funcs[data_type]

        self.db_path=db_path

        cache_file=os.path.join(CACHE_FOLDER,map_name(db_path))

        if USE_CACHE and os.path.exists(cache_file):
            logging.info(f"using local cache {cache_file}")
            self.db_path=cache_file

        if prefix is None:
            self.prefix = None
        else:
            self.prefix=f"{prefix}_"
        
        self._init=False
        preload(self,load=load)

    def initialize(self):
        self.env = lmdb.open(
            self.db_path,
            subdir=osp.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=10000
        )
        

        self._init=True
        
    @property
    def keys(self):
        while not self._init:
            time.sleep(0.1)
        if self.prefix is not None:
            _keys=[self.prefix+key.decode() for key in get_keys_from_lmdb(self.env)]
        else:
            _keys=[key.decode() for key in get_keys_from_lmdb(self.env)]
        return _keys

    def get(self, name):
        env = self.env
        if self.prefix is not None:
            name=remove_prefix(name,self.prefix)
        with env.begin(write=False) as txn:
            byteflow = txn.get(name.encode())
        if byteflow is None:
            print("fuck",name)
            raise name
        return self.decode_func(byteflow)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.env


import json

class KVReader:
    def __init__(self,db_path,data_type,prefix=None,load=True):
        assert data_type=="text"        
        if prefix is None:
            self.prefix = None
        else:
            self.prefix=f"{prefix}_"
        self.db_path=db_path
        preload(self,load=load)
        self._init=False
        self._opened=False

    def initialize(self):
        f=open(self.db_path,"r")
        start=int(f.read(1000).strip())
        f.seek(start)
        self.mp=json.load(f)
        if self.prefix is not None:
            self.mp={self.prefix+k:v for k,v in self.mp.items()}
        f.close()
        self._init=True
    
    def open(self):
        self.f=open(self.db_path,"r")
        self._opened=True
        
    @property
    def keys(self):
        while not self._init:
            time.sleep(0.1)
        return list(self.mp.keys())
        
    def get(self,key):
        if not self._opened:
            self.open()
        idx=self.mp[key]
        self.f.seek(idx)
        text=self.f.readline().strip()
        return {"alt_text":text}
    
    def __len__(self):
        return len(self.mp)
    
    @staticmethod
    def create(file,keys,values):
        assert len(keys)==len(values)
        f=open(file,"w")
        f.write("\n"*1000)
        idx=[]
        for val in values:
            idx.append(f.tell())
            f.write(val)
            f.write("\n")
        start=f.tell()
        ki={k:i for i,k in zip(idx,keys)}
        json.dump(ki, f, ensure_ascii=False)
        f.seek(0)
        f.write(str(start))
        f.close()
        



class MultipleLMDBManager:
    def __init__(self, files: list, data_type,get_key=False,sync=True):
        self.files = files
        self._is_init = False
        self.data_type=data_type
        assert data_type in decode_funcs
        self.get_key=get_key

        if sync:
            print("sync",files)
            self.initialize()
        else:
            print("async",files)
            preload(self)

    def keep_subset(self,subset):
        mapping={key:self.mapping[key] for key in subset}
        del self.mapping
        self.mapping=mapping


    def initialize(self):
        self.mapping={}
        self.managers={}
        new_files=[]
        for old_file in self.files:
            items=old_file.split("|")
            file=items[0]
            if len(items)>1:
                prefix = items[1]
            else:
                prefix = None
            if not file.startswith("glob-"):
                new_files.append(old_file)
            else:
                desc=remove_prefix(file,"glob-")
                sub_files = glob.glob(desc)
                sub_files = sorted(sub_files)
                if prefix is not None:
                    sub_files = [f"{f}|{prefix}" for f in sub_files]
                new_files.extend(sub_files)

        self.files=new_files

        for i,old_file in enumerate(self.files):
            items=old_file.split("|")
            file=items[0]
            if len(items)>1:
                prefix = items[1]
            else:
                prefix = None
            if file.endswith(".lmdb"):
                Manager = DBManager
            elif file.endswith(".zip"):
                Manager = ZipManager
            elif file.endswith(".kv"):
                Manager = KVReader
            else:
                raise 
            self.managers[i] = Manager(file,self.data_type,prefix=prefix,load=False)
            print(file, " done")

        ThreadPool(4).run(preload,self.managers.values()) 
        

        if self.get_key:
            self._keys=[]
            for index,manager in self.managers.items():
                file=manager.db_path
                print(f"{file} loading")
                logging.info(f"{file} loading")
                keys=manager.keys
                self._keys.extend(keys)
                for key in keys:
                    self.mapping[key]=index
                logging.info(f"{file} loaded, size = {len(keys)}")
                print(f"{file} loaded, size = {len(keys)}")

        self._is_init=True


    @property
    def keys(self):
        while not self._is_init:
            time.sleep(0.1)
        return self._keys

    def cleanup(self):
        del self._keys
        del self.mapping

    def get(self, name,source=None):
        if source is None:
            source=self.mapping[name]
        data = self.managers[source].get(name)
        return data




class MetaDB:
    def __init__(self, path, readonly=True, size=None):
        self.readonly = readonly
        if readonly:
            self.db = lmdb.open(
                path, readonly=readonly, max_readers=10000, subdir=False, lock=False
            )
        else:
            assert size is not None
            self.db = lmdb.open(
                path,
                readonly=readonly,
                max_readers=10000,
                subdir=False,
                map_size=int(1073741824 * size),
            )

    def keys(self):
        with self.db.begin(write=False) as txn:
            keys = list(txn.cursor().iternext(values=False))
        return keys

    def encode_int(self,num):
        return num.to_bytes(4,"big")

    def decode_int(self,num_bytes):
        return int.from_bytes(num_bytes,"big")

    def get(self, key, func=None):
        with self.db.begin(write=False) as txn:
            val = txn.get(key)
            if val is None:
                raise
            if func:
                val = func(val)
            return val


