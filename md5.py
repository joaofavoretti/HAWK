# Considering that only whole bytes are going to be used
# RFC: https://datatracker.ietf.org/doc/html/rfc1321

import numpy as np

def md5(fname):
    with open(fname, 'rb') as f:
        return _md5_calc(f.read())

def _md5_calc(msg):

    pad_msg = _pad_msg(msg)

    # Initialize the four word buffers
    A = np.uint32(0x67452301)
    B = np.uint32(0xEFCDAB89)
    C = np.uint32(0x98BADCFE)
    D = np.uint32(0x10325476)

    # Create the T vector properly
    T = np.zeros(64, dtype=np.uint32)
    for i in range(64):
        T[i] = np.uint32(2**32 * abs(np.sin(i + 1)))

    # Process each 16-word block
    for i in range(0, len(pad_msg), 64):
        # Break the block into 16 words
        X = np.zeros(16, dtype=np.uint32)
        for j in range(16):
            # CHECK: Isso pode ser um ponto de troca de endianness se tiver errado
            X[j] = np.uint32(pad_msg[i + j * 4]) \
                | np.uint32(pad_msg[i + j * 4 + 1]) << 8 \
                | np.uint32(pad_msg[i + j * 4 + 2]) << 16 \
                | np.uint32(pad_msg[i + j * 4 + 3]) << 24

        # Save the current state
        AA = A
        BB = B
        CC = C
        DD = D

        # Round 1
        def _round1(a, b, c, d, k, s, i):
            return b + np.uint32((a + _F(b, c, d) + X[k] + T[i - 1]) << s)
        
        A = _round1(A, B, C, D, 0, 7, 1); D = _round1(D, A, B, C, 1, 12, 2); C = _round1(C, D, A, B, 2, 17, 3); B = _round1(B, C, D, A, 3, 22, 4)
        A = _round1(A, B, C, D, 4, 7, 5); D = _round1(D, A, B, C, 5, 12, 6); C = _round1(C, D, A, B, 6, 17, 7); B = _round1(B, C, D, A, 7, 22, 8)
        A = _round1(A, B, C, D, 8, 7, 9); D = _round1(D, A, B, C, 9, 12, 10); C = _round1(C, D, A, B, 10, 17, 11); B = _round1(B, C, D, A, 11, 22, 12)
        A = _round1(A, B, C, D, 12, 7, 13); D = _round1(D, A, B, C, 13, 12, 14); C = _round1(C, D, A, B, 14, 17, 15); B = _round1(B, C, D, A, 15, 22, 16)

        # Round 2
        def _round2(a, b, c, d, k, s, i):
            return b + np.uint32((a + _G(b, c, d) + X[k] + T[i - 1]) << s)

        A = _round2(A, B, C, D, 1, 5, 17); D = _round2(D, A, B, C, 6, 9, 18); C = _round2(C, D, A, B, 11, 14, 19); B = _round2(B, C, D, A, 0, 20, 20)
        A = _round2(A, B, C, D, 5, 5, 21); D = _round2(D, A, B, C, 10, 9, 22); C = _round2(C, D, A, B, 15, 14, 23); B = _round2(B, C, D, A, 4, 20, 24)
        A = _round2(A, B, C, D, 9, 5, 25); D = _round2(D, A, B, C, 14, 9, 26); C = _round2(C, D, A, B, 3, 14, 27); B = _round2(B, C, D, A, 8, 20, 28)
        A = _round2(A, B, C, D, 13, 5, 29); D = _round2(D, A, B, C, 2, 9, 30); C = _round2(C, D, A, B, 7, 14, 31); B = _round2(B, C, D, A, 12, 20, 32)

        # Round 3
        def _round3(a, b, c, d, k, s, i):
            return b + np.uint32((a + _H(b, c, d) + X[k] + T[i - 1]) << s)

        A = _round3(A, B, C, D, 5, 4, 33); D = _round3(D, A, B, C, 8, 11, 34); C = _round3(C, D, A, B, 11, 16, 35); B = _round3(B, C, D, A, 14, 23, 36)
        A = _round3(A, B, C, D, 1, 4, 37); D = _round3(D, A, B, C, 4, 11, 38); C = _round3(C, D, A, B, 7, 16, 39); B = _round3(B, C, D, A, 10, 23, 40)
        A = _round3(A, B, C, D, 13, 4, 41); D = _round3(D, A, B, C, 0, 11, 42); C = _round3(C, D, A, B, 3, 16, 43); B = _round3(B, C, D, A, 6, 23, 44)
        A = _round3(A, B, C, D, 9, 4, 45); D = _round3(D, A, B, C, 12, 11, 46); C = _round3(C, D, A, B, 15, 16, 47); B = _round3(B, C, D, A, 2, 23, 48)

        # Round 4
        def _round4(a, b, c, d, k, s, i):
            return b + np.uint32((a + _I(b, c, d) + X[k] + T[i - 1]) << s)

        A = _round4(A, B, C, D, 0, 6, 49); D = _round4(D, A, B, C, 7, 10, 50); C = _round4(C, D, A, B, 14, 15, 51); B = _round4(B, C, D, A, 5, 21, 52)
        A = _round4(A, B, C, D, 12, 6, 53); D = _round4(D, A, B, C, 3, 10, 54); C = _round4(C, D, A, B, 10, 15, 55); B = _round4(B, C, D, A, 1, 21, 56)
        A = _round4(A, B, C, D, 8, 6, 57); D = _round4(D, A, B, C, 15, 10, 58); C = _round4(C, D, A, B, 6, 15, 59); B = _round4(B, C, D, A, 13, 21, 60)
        A = _round4(A, B, C, D, 4, 6, 61); D = _round4(D, A, B, C, 11, 10, 62); C = _round4(C, D, A, B, 2, 15, 63); B = _round4(B, C, D, A, 9, 21, 64)

        # Add the saved state to the current state, considering a wrap around if overflows
        A = np.uint32(A + AA)
        B = np.uint32(B + BB)
        C = np.uint32(C + CC)
        D = np.uint32(D + DD)

    # Convert the four words to bytes
    out = bytes()
    out += _to_bytes(A)
    out += _to_bytes(B)
    out += _to_bytes(C)
    out += _to_bytes(D)

    return out

def _to_bytes(num):
    assert isinstance(num, np.uint32)
    
    # CHECK: Potentially a missing endianess swap
    out = bytes([(num >> 24) & 0xFF])
    out += bytes([(num >> 16) & 0xFF])
    out += bytes([(num >> 8) & 0xFF])
    out += bytes([num & 0xFF])

    return out

def _pad_msg(msg):

    msg_len = len(msg)
    
    # Calculate the padding
    pad_len = 56 - (msg_len % 64)
    if pad_len <= 0:
        pad_len += 64
    
    pad = bytes([0b10000000]) + bytes([0b00000000]) * (pad_len - 1)
    pad += _msg_size_bytes(msg_len)

    return msg + pad

def _msg_size_bytes(num):
    high_word = (num >> 32) & 0xFFFFFFFF
    low_word = num & 0xFFFFFFFF

    # CHECK: Isso pode ser um ponto de troca de endianess
    return low_word.to_bytes(4, 'big') + high_word.to_bytes(4, 'big')

def _F(x, y, z):
    # Verify the types of x, y, and z
    assert isinstance(x, np.uint32)
    assert isinstance(y, np.uint32)
    assert isinstance(z, np.uint32)

    # CHECK: Verificar se as operações de bitwise sao executadas corretamente com os tipos
    #  do numpy
    return (x & y) | (~x & z)

def _G(x, y, z):
    # Verify the types of x, y, and z
    assert isinstance(x, np.uint32)
    assert isinstance(y, np.uint32)
    assert isinstance(z, np.uint32)

    return (x & z) | (y & ~z)

def _H(x, y, z):
    # Verify the types of x, y, and z
    assert isinstance(x, np.uint32)
    assert isinstance(y, np.uint32)
    assert isinstance(z, np.uint32)

    return x ^ y ^ z

def _I(x, y, z):
    # Verify the types of x, y, and z
    assert isinstance(x, np.uint32)
    assert isinstance(y, np.uint32)
    assert isinstance(z, np.uint32)

    return y ^ (x | ~z)