import argparse
from md5 import md5


FUNCTION_BIND = {
    'md5': md5
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', help='File to hash', required=True)
    parser.add_argument('--alg', '-a', help='Hashing algorithm', choices=['md5'], default='md5')
    
    args = parser.parse_args()

    print(FUNCTION_BIND[args.alg](args.file).hex())
    
if __name__ == '__main__':
    main()