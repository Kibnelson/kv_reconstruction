// CPU reference acceleration for the frozen LKVR v1 format. No format changes.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {
uint32_t u32(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1])<<8) | (uint32_t(p[2])<<16) | (uint32_t(p[3])<<24); }
uint32_t bits(const uint8_t* p, size_t bit, unsigned width) {
    uint32_t result=0;
    for (unsigned b=0;b<width;++b) result |= uint32_t((p[(bit+b)/8] >> ((bit+b)%8))&1) << b;
    return result;
}
int fail(char* error, size_t n, const char* message) { if (n) std::snprintf(error,n,"%s",message); return 1; }
}

extern "C" int lkv_decode_body(const uint8_t* data, size_t length,
    const uint16_t* codebook, size_t centroids, unsigned qbits,
    size_t total, size_t group_size, uint16_t* output, char* error, size_t error_size) {
    if (!group_size || group_size%8 || group_size<8 || (qbits!=4&&qbits!=6&&qbits!=8) || centroids!=(size_t(1)<<qbits))
        return fail(error,error_size,"invalid native decoder arguments");
    size_t offset=0;
    for (size_t start=0;start<total;) {
        if (length-offset<5) return fail(error,error_size,"truncated learned group");
        unsigned mode=data[offset]; size_t payload_length=u32(data+offset+1); offset+=5;
        if (payload_length>length-offset) return fail(error,error_size,"truncated learned group payload");
        const uint8_t* p=data+offset; offset+=payload_length;
        const size_t count=group_size<total-start?group_size:total-start;
        if (mode==0) {
            if (count>std::numeric_limits<size_t>::max()/2 || payload_length!=2*count)
                return fail(error,error_size,"incorrect raw group length");
            for (size_t i=0;i<count;++i) output[start+i]=uint16_t(p[2*i]) | (uint16_t(p[2*i+1])<<8);
        } else if (mode==1) {
            const size_t nvec=count/8+(count%8!=0);
            const size_t qlength=(nvec*qbits+7)/8, wlength=(nvec*4+7)/8, slength=nvec;
            if (qlength+wlength+slength>payload_length)
                return fail(error,error_size,"truncated learned control codes");
            const uint8_t *widths=p+qlength,*signs=widths+wlength,*corrections=signs+slength;
            size_t expected=qlength+wlength+slength;
            for (size_t v=0;v<nvec;++v) expected+=bits(widths,v*4,4);
            if (expected!=payload_length) return fail(error,error_size,"incorrect correction length");
            size_t correction_offset=0;
            for (size_t v=0;v<nvec;++v) {
                const uint32_t index=bits(p,v*qbits,qbits), width=bits(widths,v*4,4);
                if (index>=centroids) return fail(error,error_size,"invalid codebook index");
                const uint32_t modulus=uint32_t(1)<<width, half=width?modulus/2:0;
                for (size_t d=0;d<8;++d) {
                    const uint32_t residue=bits(corrections+correction_offset,d*width,width);
                    const int32_t prediction=codebook[index*8+d];
                    const int32_t recovered=prediction+int32_t((residue-uint32_t(prediction)+half)&(modulus-1))-int32_t(half);
                    if (recovered<0 || recovered>0x7fff) return fail(error,error_size,"decoded BF16 magnitude outside range");
                    if (v*8+d<count) output[start+v*8+d]=uint16_t(recovered) | (uint16_t((signs[v]>>d)&1)<<15);
                }
                correction_offset+=width; // Eight values times width bits = width bytes.
            }
        } else return fail(error,error_size,"unknown learned group mode");
        start+=count;
    }
    if (offset!=length) return fail(error,error_size,"unexpected trailing learned bytes");
    return 0;
}
