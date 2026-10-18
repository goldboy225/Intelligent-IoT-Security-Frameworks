"""Fetch public CSV/ZIP data only; verify pinned mirror content hashes."""
import hashlib
from pathlib import Path
import requests

SOURCES=[
    ('ML-EdgeIIoT-dataset.csv',
     'https://raw.githubusercontent.com/arushn47/edgeiiot-data-analysis/87369dee96e284e7b7fcd9c733534f7ab8f90166/ML-EdgeIIoT-dataset.csv',
     '53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a'),
    ('Bot-IoT-mirror.zip',
     'https://raw.githubusercontent.com/AG24-hub/Gradient-Boosting-algorithm-on-BOT-IOT-Dataset/8b8fdcce976af3dad10e914fe97cb9f6e56c0af7/Dataset.zip',
     '248b99eb65015b03069ecccf0421b48fba4d762ab92e9ed5bb7e434c06242bd8'),
]


def main():
    root=Path('/workspace/edgeguard/data');root.mkdir(parents=True,exist_ok=True)
    for filename,url,expected in SOURCES:
        path=root/filename
        if not path.exists():
            response=requests.get(url,timeout=180);response.raise_for_status()
            content=response.content
            assert hashlib.sha256(content).hexdigest()==expected,filename+' download checksum mismatch'
            path.write_bytes(content)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,filename+' local checksum mismatch'
        print(filename,'SHA-256 verified',flush=True)


if __name__=='__main__':main()
