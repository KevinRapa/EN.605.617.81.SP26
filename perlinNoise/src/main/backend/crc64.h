
#ifndef CRC64_H
#define CRC64_H

typedef unsigned long long DWORD64;

/**
 * CRC64 function
 *
 * @param[in] data pointer to the buffer
 * @param[in] len length of buffer in bytes
 * @param[in,out] hash in = initial value of the hash; out = returned hash
 */
void GetCRC64(DWORD64 &crc, const unsigned char *cp, unsigned long length);

#endif
