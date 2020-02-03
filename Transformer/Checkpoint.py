from pathlib import Path
import re
import tensorflow as tf

def get_epoch(fname) :
    result = re.findall("epoch-\d+", fname.lower())

    if len(result) :
        return int(result[0].split('-')[1])
    else :
        print("\n{}에서 'epoch-\\d+' 형식의 epoch를 찾지 못했습니다.".format(fname))
        return 0

def find_checkpoint_file(ckpt_path, epoch) :
    ckpt_files = [str(fname) for fname in ckpt_path.glob("*")]
    ckpt_files.sort(reverse = True)

    ckpt = None
    if epoch is not None :
        ckpt_prefix = str(ckpt_path / "Epoch-{}".format(epoch))
        initial_epoch = epoch

        for _file in ckpt_files :
            if _file.startswith(ckpt_prefix) :
                ckpt = str(Path(_file).parent / Path(_file).stem) # 확장자 부분 제외
                break

        if ckpt is None :
            raise FileNotFoundError("\n{} checkpoint를 찾을 수 없음\n".format(ckpt_prefix))

    else : # 가장 최근의 checkpoint를 불러옴
        ckpt = tf.train.latest_checkpoint(ckpt_path)
        initial_epoch = get_epoch(Path(ckpt).name) # 파일 이름만 전달

    return ckpt, initial_epoch + 1

def load_checkpoint(ckpt_path, ckpt_epoch) :
    if ckpt_path.is_dir() and not ckpt_path.exists() :
        raise FileNotFoundError("\n{} 경로를 찾을 수 없음\n".format(ckpt_path))

    if ckpt_path.is_dir() :
        ckpt_path = ckpt_path / "ckpt"
        return find_checkpoint_file(ckpt_path, ckpt_epoch)
    else : # File
        return str(ckpt_path), get_epoch(ckpt_path.name) + 1