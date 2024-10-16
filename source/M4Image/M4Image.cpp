#include "M4Image/M4Image.h"
#include <map>
#include <stdint.h>

#include <mango/image/surface.hpp>
#include <mango/image/decoder.hpp>
#include <pixman.h>

static M4Image::MallocProc mallocProc = malloc;
static M4Image::FreeProc freeProc = free;
static M4Image::ReallocProc reallocProc = realloc;

// normally the free function will work even if it is passed a null pointer
// but since we allow custom allocators to be used, this may not always be a guarantee
// since I rely on this, I wrap the free procedure here to check
void freeSafe(unsigned char* &block) {
    if (block) {
        freeProc(block);
    }

    block = 0;
}

// the mango::core::MemoryStream class does not allow using a custom allocator
// so we must implement our own
// this is here instead of in the header so that it's private to this translation unit
class AllocatorStream : public mango::Stream {
    private:
    struct State {
        mango::u64 capacity = 0;
        mango::u64 size = 0;
        mango::u64 offset = 0;
        mango::u8* data = 0;
    };

    State state = {};

    public:
    virtual ~AllocatorStream();
    mango::u64 capacity() const;
    mango::u64 size() const;
    mango::u64 offset() const;
    mango::u8* acquire();
    void reserve(mango::u64 offset);
    void seek(mango::s64 distance, SeekMode mode);
    void read(void* dest, mango::u64 size);
    void write(const void* data, mango::u64 size);
};

AllocatorStream::~AllocatorStream() {
    // note that for acquire to work this cannot happen in State's destructor
    freeSafe(state.data);
}

mango::u64 AllocatorStream::capacity() const {
    return state.capacity;
}

mango::u64 AllocatorStream::size() const {
    return state.size;
}

mango::u64 AllocatorStream::offset() const {
    return state.offset;
}

mango::u8* AllocatorStream::acquire() {
    mango::u8* data = state.data;
    state = {};
    return data;
}

void AllocatorStream::reserve(mango::u64 offset) {
    mango::u64 capacity = state.capacity;

    if (offset <= capacity) {
        return;
    }

    if (!capacity) {
        // initial capacity is 128 bytes
        // written as 64 here because we will immediately multiply this by two
        // this number chosen because it is probably the smallest reasonable size for an image
        // (it is the exact size of a 2x2 pixel white PNG)
        // must be a power of two
        const mango::u64 INITIAL_CAPACITY = 64;

        capacity = INITIAL_CAPACITY;
    }

    // find power of two greater than or equal to the offset
    do {
        capacity *= 2;
    } while (offset > capacity);

    state.data = (mango::u8*)reallocProc(state.data, capacity);

    if (!state.data) {
        MANGO_EXCEPTION("[AllocatorStream] Failed to reallocate memory.");
    }

    state.capacity = capacity;
}

void AllocatorStream::seek(mango::s64 distance, SeekMode mode) {
    switch (mode) {
        case BEGIN:
        state.offset = distance;
        break;
        case CURRENT:
        state.offset += distance;
        break;
        case END:
        state.offset = state.size - distance;
    }
}

void AllocatorStream::read(void* dest, mango::u64 size) {
    if (!size) {
        return;
    }

    mango::u64 offset = state.offset + size;

    // ensure we didn't overflow
    if (offset < state.offset) {
        MANGO_EXCEPTION("[AllocatorStream] Size too large.");
    }

    // ensure we don't read past the end
    if (offset > state.size) {
        MANGO_EXCEPTION("[AllocatorStream] Reading past end of data.");
    }

    // we have verified that size is less than the size of the data
    // so it is safe to use as the source size
    if (memcpy_s(dest, size, state.data + state.offset, size)) {
        MANGO_EXCEPTION("[AllocatorStream] Failed to copy memory.");
    }

    state.offset = offset;
}

void AllocatorStream::write(const void* data, mango::u64 size) {
    if (!size) {
        return;
    }

    mango::u64 offset = state.offset + size;

    // ensure we didn't overflow
    if (offset < state.offset) {
        MANGO_EXCEPTION("[AllocatorStream] Size too large.");
    }

    // ensure we don't write past the end
    reserve(offset);

    // zero seeked over data
    if (state.offset > state.size) {
        memset(state.data + state.size, 0, state.offset - state.size);
    }

    if (memcpy_s(state.data + state.offset, state.capacity - state.offset, data, size)) {
        MANGO_EXCEPTION("[AllocatorStream] Failed to copy memory.");
    }

    // set the size
    if (offset > state.size) {
        state.size = offset;
    }

    state.offset = offset;
}

typedef std::map<M4Image::COLOR_FORMAT, mango::image::Format> COLOR_FORMAT_MAP;

static const COLOR_FORMAT_MAP FORMAT_MAP = {
    {M4Image::COLOR_FORMAT::RGBA32, mango::image::Format(32, mango::image::Format::UNORM, mango::image::Format::RGBA, 8, 8, 8, 8)},
    {M4Image::COLOR_FORMAT::RGBX32, mango::image::Format(32, mango::image::Format::UNORM, mango::image::Format::RGBA, 8, 8, 8, 0)},
    {M4Image::COLOR_FORMAT::BGRA32, mango::image::Format(32, mango::image::Format::UNORM, mango::image::Format::BGRA, 8, 8, 8, 8)},
    {M4Image::COLOR_FORMAT::BGRX32, mango::image::Format(32, mango::image::Format::UNORM, mango::image::Format::BGRA, 8, 8, 8, 0)},
    {M4Image::COLOR_FORMAT::RGB24, mango::image::Format(24, mango::image::Format::UNORM, mango::image::Format::RGB, 8, 8, 8, 0)},
    {M4Image::COLOR_FORMAT::BGR24, mango::image::Format(24, mango::image::Format::UNORM, mango::image::Format::BGR, 8, 8, 8, 0)},
    {M4Image::COLOR_FORMAT::AL16, mango::image::LuminanceFormat(16, mango::image::Format::UNORM, 8, 8)},
    {M4Image::COLOR_FORMAT::A8, mango::image::Format(8, mango::image::Format::UNORM, mango::image::Format::A, 8, 0, 0, 0)},
    {M4Image::COLOR_FORMAT::L8, mango::image::LuminanceFormat(8, mango::image::Format::UNORM, 8, 0)},
    {M4Image::COLOR_FORMAT::XXXL32, mango::image::LuminanceFormat(32, 0xFF000000, 0x00000000)},
    {M4Image::COLOR_FORMAT::XXLA32, mango::image::LuminanceFormat(32, 0x00FF0000, 0xFF000000)},
};

void decodeSurfaceImage(mango::image::Surface &surface, mango::image::ImageDecoder &imageDecoder) {
    // allocate memory for the image
    surface.image = (mango::u8*)mallocProc(surface.stride * (size_t)surface.height);

    if (!surface.image) {
        return;
    }

    MAKE_SCOPE_EXIT(surfaceImageScopeExit) {
        freeSafe(surface.image);
    };

    // uncomment the second argument to disable multithreading
    mango::image::ImageDecodeStatus status = imageDecoder.decode(surface/*, {nullptr, true, false}*/);

    // status is false if decoding the image failed
    if (!status) {
        MANGO_EXCEPTION("[INFO] {}", status.info);
    }

    surfaceImageScopeExit.dismiss();
}

unsigned char* encodeSurfaceImage(const mango::image::Surface &surface, const char* extension, size_t &size, float quality = 0.90f) {
    AllocatorStream allocatorStream = AllocatorStream();
    mango::image::ImageEncodeStatus status = surface.save(allocatorStream, extension, { {}, {}, quality });

    if (!status) {
        MANGO_EXCEPTION("[INFO] {}", status.info);
    }

    size = allocatorStream.size();
    return allocatorStream.acquire();
}

void blitSurfaceImage(const mango::image::Surface &inputSurface, mango::image::Surface &outputSurface) {
    size_t outputSurfaceImageSize = outputSurface.stride * (size_t)outputSurface.height;
    outputSurface.image = (mango::u8*)mallocProc(outputSurfaceImageSize);

    if (!outputSurface.image) {
        return;
    }

    MAKE_SCOPE_EXIT(outputSurfaceImageScopeExit) {
        freeSafe(outputSurface.image);
    };

    if (
        inputSurface.format == outputSurface.format
        && inputSurface.stride == outputSurface.stride
        && inputSurface.width == outputSurface.width
        && inputSurface.height == outputSurface.height
    ) {
        if (memcpy_s(outputSurface.image, outputSurfaceImageSize, inputSurface.image, inputSurface.stride * (size_t)inputSurface.height)) {
            return;
        }
    } else {
        outputSurface.blit(0, 0, inputSurface);
    }
    outputSurfaceImageScopeExit.dismiss();
}

typedef std::map<M4Image::COLOR_FORMAT, pixman_format_code_t> PIXMAN_FORMAT_CODE_MAP;

// no, these are not backwards (read comments for getResizeFormat function below)
static const PIXMAN_FORMAT_CODE_MAP RGBA_PIXMAN_FORMAT_CODE_MAP = {
    {M4Image::COLOR_FORMAT::RGBA32, PIXMAN_a8r8g8b8},
    {M4Image::COLOR_FORMAT::RGBX32, PIXMAN_x8r8g8b8},
    {M4Image::COLOR_FORMAT::BGRA32, PIXMAN_a8b8g8r8},
    {M4Image::COLOR_FORMAT::BGRX32, PIXMAN_x8b8g8r8},
    {M4Image::COLOR_FORMAT::RGB24, PIXMAN_r8g8b8},
    {M4Image::COLOR_FORMAT::BGR24, PIXMAN_b8g8r8},
    {M4Image::COLOR_FORMAT::A8, PIXMAN_a8}
};

static const PIXMAN_FORMAT_CODE_MAP BGRA_PIXMAN_FORMAT_CODE_MAP = {
    {M4Image::COLOR_FORMAT::RGBA32, PIXMAN_a8b8g8r8},
    {M4Image::COLOR_FORMAT::RGBX32, PIXMAN_x8b8g8r8},
    {M4Image::COLOR_FORMAT::BGRA32, PIXMAN_a8r8g8b8},
    {M4Image::COLOR_FORMAT::BGRX32, PIXMAN_x8r8g8b8},
    {M4Image::COLOR_FORMAT::RGB24, PIXMAN_b8g8r8},
    {M4Image::COLOR_FORMAT::BGR24, PIXMAN_r8g8b8},
    {M4Image::COLOR_FORMAT::A8, PIXMAN_a8}
};

// the COLOR_FORMAT enum uses little endian, like mango
// note that Pixman expresses colours in big endian (the opposite of mango)
// so Pixman's "ARGB" is equal to mango's BGRA
// this makes the constants a little confusing
// from here on out, I'll be using little endian (like mango)
M4Image::COLOR_FORMAT getResizeColorFormat(
    bool rgba,
    M4Image::COLOR_FORMAT colorFormat,
    pixman_format_code_t &sourceFormat,
    pixman_format_code_t &destinationFormat
) {
    switch (colorFormat) {
        case M4Image::COLOR_FORMAT::A8:
        // for COLOR_FORMAT::A8, the colours do not need to be premultiplied
        // because we toss them anyway, only keeping the alpha
        // we still use a 32-bit source format to keep things fast
        // only converting to an 8-bit format at the end, as is typical
        // note the break - we don't return here, because
        // the input format is interchangeable between RGBA/BGRA
        sourceFormat = PIXMAN_a8r8g8b8;
        break;
        case M4Image::COLOR_FORMAT::L8:
        // for COLOR_FORMAT::L8, the image is loaded in XXXL format
        // so Pixman will think the luminance is "alpha"
        // the destination is set to PIXMAN_a8, tossing the reserved channels
        // this once again allows us to keep the source format 32-bit
        sourceFormat = PIXMAN_a8r8g8b8;
        destinationFormat = PIXMAN_a8;
        return M4Image::COLOR_FORMAT::XXXL32;
        case M4Image::COLOR_FORMAT::XXXL32:
        // here we just keep it 32-bit the whole way, easy peasy
        // we do the same trick as COLOR_FORMAT::L8 where luminance = Pixman's alpha
        sourceFormat = PIXMAN_a8r8g8b8;
        destinationFormat = PIXMAN_a8r8g8b8;
        return M4Image::COLOR_FORMAT::XXXL32;
        case M4Image::COLOR_FORMAT::AL16:
        case M4Image::COLOR_FORMAT::XXLA32:
        // for COLOR_FORMAT::AL16, mango reads the image in XXLA format
        // the luminance can't be in the alpha channel now
        // because we need to premultiply in this case
        // so we just shove it in the blue channel now
        // Pixman does not support any 16-bit formats with
        // two 8-bit channels, so
        // we manually convert down to 16-bit during the
        // unpremultiplication step, since we need to do a whole
        // pass over the image for that anyway, so might as well
        // kill two birds with one stone
        sourceFormat = PIXMAN_x8r8g8b8;
        destinationFormat = PIXMAN_a8r8g8b8;
        return M4Image::COLOR_FORMAT::XXLA32;
        default:
        // sourceFormat is only ever PIXMAN_x8r8g8b8 or PIXMAN_a8r8g8b8
        // it is the former if the colours will need to be premultiplied
        // and the latter if they do not need to be
        // it is mango's job to decode the image into a 32-bit format before
        // ever being passed to Pixman, which is only fast with 32-bit colour
        sourceFormat = PIXMAN_x8r8g8b8;
    }

    // as an optimization, we allow mango to import in RGBA
    // RGBA and BGRA are the only import formats allowed
    // (A must come last, and these are the only formats Pixman supports where A is last)
    if (rgba) {
        // for all other colour formats, we start with 32-bit colour, converting down as necessary
        // Pixman operates the fastest in BGRA mode, but since both operations
        // that we are doing apply the same to all colour channels (premultiplying and interpolating)
        // the only thing that matters is the position of the alpha channel, so
        // we can just "lie" to Pixman and say our RGBA image is BGRA
        // then, flip it to "RGBA" at the end if BGRA was requested
        destinationFormat = RGBA_PIXMAN_FORMAT_CODE_MAP.at(colorFormat);
        return M4Image::COLOR_FORMAT::RGBA32;
    }

    // once again, these are not wrong. Setting ARGB as the destination
    // means to "flip" the colour, as Pixman always thinks the image is ABGR
    destinationFormat = BGRA_PIXMAN_FORMAT_CODE_MAP.at(colorFormat);
    return M4Image::COLOR_FORMAT::BGRA32;
}

// if we need to premultiply the colours
// then we use a mask (from which only the alpha channel is used)
// on the surface (which we mark as RGBX, so its alpha is ignored)
// this has the effect of multiplying RGB * Alpha, which
// is the exact operation we need
// (doing it this way is significantly more performant than manually doing it ourselves)
// both use the same underlying data, so no memory is allocated
// they are just seperate views into the same memory
// note that we need the alpha channel for the actual resize
// so we use maskImage as the destination, rather than writing over sourceImage
// therefore the sourceImage (which is RGBX) is only used for this premultiply step
// and the maskImage is what is what we actually transform later on
// if you're totally lost on why this is needed for resizing, then
// see: https://www.realtimerendering.com/blog/gpus-prefer-premultiplication/
pixman_image_t* premultiplyMaskImage(const mango::image::Surface &surface, pixman_image_t* sourceImage) {
    pixman_image_t* maskImage = pixman_image_create_bits(
        PIXMAN_a8r8g8b8,
        surface.width, surface.height,
        (uint32_t*)surface.image,
        (int)surface.stride
    );

    if (!maskImage) {
        return 0;
    }

    MAKE_SCOPE_EXIT(maskImageScopeExit) {
        unrefImage(maskImage);
    };

    pixman_image_composite(
        PIXMAN_OP_SRC,
        sourceImage, maskImage, maskImage,
        0, 0, 0, 0, 0, 0,
        surface.width, surface.height
    );

    maskImageScopeExit.dismiss();
    return maskImage;
}

// Pixman wants a scale (like a percentage to resize by,) not a pixel size
// so here we create that
bool setTransform(pixman_image_t* maskImage, const mango::image::Surface &surface, int width, int height) {
    // this is initialized by pixman_transform_init_identity
    pixman_transform_t transform;
    pixman_transform_init_identity(&transform);

    double sx = (double)surface.width / width;
    double sy = (double)surface.height / height;

    if (!pixman_transform_scale(&transform, NULL, pixman_double_to_fixed(sx), pixman_double_to_fixed(sy))) {
        return false;
    }

    if (!pixman_image_set_transform(maskImage, &transform)) {
        return false;
    }
    return true;
}

// aligned to nearest 64 bytes so it is on cache lines
static bool channelUnpremultiplierCreated = false;
__declspec(align(64)) static unsigned char CHANNEL_UNPREMULTIPLIER[USHRT_MAX + 1] = {};

void createChannelUnpremultiplier() {
    if (channelUnpremultiplierCreated) {
        return;
    }

    // note: the alpha, divided by two, is added to the channel
    // so the channel is scaled instead of stripped (it works out to rounding the number, instead of flooring)
    // alpha starts at one, since if it's zero the colour is invisible anyway (and thus would be a divide by zero)
    const size_t DIVIDE_BY_TWO = 1;

    for (int channel = 0; channel <= UCHAR_MAX; channel++) {
        for (int alpha = 1; alpha <= UCHAR_MAX; alpha++) {
            CHANNEL_UNPREMULTIPLIER[(channel << CHAR_BIT) | alpha] = clampUCHAR(((channel * UCHAR_MAX) + (alpha >> DIVIDE_BY_TWO)) / alpha);
        }
    }

    // this function may get hit by multiple threads at once
    // however, it's entirely deterministic, so it doesn't really matter
    // the worst case scenario is that one thread hits this as
    // another is just finishing
    // but even in that case, it only costs the same time as processing a
    // single 255 * 255 image the slow way, two times, at most
    // which doesn't actually take very long, usually only 1 ms
    // (and then processing larger images costs nothing, pennies, so it's extremely worth it)
    // having a lock here would just require threads to wait about the
    // same amount of time for the first to finish
    // but would be a slowdown the other 99% of the time
    // so it's better to just eat the cost of needing to potentially do this multiple times
    // that said, it's important this is only set to true here at the end!
    channelUnpremultiplierCreated = true;
}

#define UNPREMULTIPLY_CHANNEL(channel, alpha) (CHANNEL_UNPREMULTIPLIER[((channel) << CHAR_BIT) | (alpha)])

unsigned char* convertImage(M4Image::Color32* colorPointer, size_t width, size_t height, size_t stride, bool unpremultiply) {
    const size_t CHANNEL_LUMINANCE = 2;
    const size_t CHANNEL_ALPHA = 3;

    unsigned char* bits = (unsigned char*)mallocProc((size_t)height * stride);

    if (!bits) {
        return 0;
    }

    MAKE_SCOPE_EXIT(bitsScopeExit) {
        freeSafe(bits);
    };

    unsigned char* rowPointer = bits;
    M4Image::Color16* luminancePointer = (M4Image::Color16*)rowPointer;

    if (unpremultiply) {
        createChannelUnpremultiplier();

        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                unsigned char &alpha = luminancePointer->channels[1];
                alpha = colorPointer->channels[CHANNEL_ALPHA];

                if (alpha) {
                    luminancePointer->channels[0] = UNPREMULTIPLY_CHANNEL(colorPointer->channels[CHANNEL_LUMINANCE], alpha);
                }

                colorPointer++;
                luminancePointer++;
            }

            rowPointer += stride;
            luminancePointer = (M4Image::Color16*)rowPointer;
        }
    } else {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                *luminancePointer++ = *(M4Image::Color16*)&colorPointer++->channels[CHANNEL_LUMINANCE];
            }

            rowPointer += stride;
            luminancePointer = (M4Image::Color16*)rowPointer;
        }
    }

    bitsScopeExit.dismiss();
    return bits;
}

void unpremultiplyColors(M4Image::Color32* colorPointer, size_t width, size_t height, size_t stride) {
    createChannelUnpremultiplier();

    const size_t CHANNEL_ALPHA = 3;

    unsigned char* rowPointer = (unsigned char*)colorPointer;

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            const unsigned char &ALPHA = colorPointer->channels[CHANNEL_ALPHA];

            if (ALPHA) {
                colorPointer->channels[0] = UNPREMULTIPLY_CHANNEL(colorPointer->channels[0], ALPHA);
                colorPointer->channels[1] = UNPREMULTIPLY_CHANNEL(colorPointer->channels[1], ALPHA);
                colorPointer->channels[2] = UNPREMULTIPLY_CHANNEL(colorPointer->channels[2], ALPHA);
            }

            colorPointer++;
        }

        rowPointer += stride;
        colorPointer = (M4Image::Color32*)rowPointer;
    }
}

unsigned char* resizeImage(
    mango::image::Surface &surface,
    int width,
    int height,
    size_t &stride,
    bool convert,
    bool isAlpha,
    pixman_format_code_t sourceFormat,
    pixman_format_code_t destinationFormat
) {
    // otherwise we have to do a bunch of boilerplate setup stuff for the resize operation
    SCOPE_EXIT {
        freeSafe(surface.image);
    };

    const size_t BYTES = 3;

    // bits MUST be a seperate buffer to surface.image (yes, even for upscaling)
    // if we aren't converting, we expect to get the destination image in the user requested stride
    // if we are converting, we expect it to be based on the format, for convenience's sake
    size_t bitsStride = (stride && !convert) ? stride : (PIXMAN_FORMAT_BPP(destinationFormat) >> BYTES) * (size_t)width;
    unsigned char* bits = (unsigned char*)mallocProc(bitsStride * (size_t)height);

    if (!bits) {
        return 0;
    }

    MAKE_SCOPE_EXIT(bitsScopeExit) {
        freeSafe(bits);
    };

    // create the destination image in the user's desired format
    // (unless we need to convert to 16-bit after, then we still make it 32-bit)
    pixman_image_t* destinationImage = pixman_image_create_bits(
        destinationFormat,
        width, height,
        (uint32_t*)bits,
        (int)bitsStride
    );

    if (!destinationImage) {
        return 0;
    }

    SCOPE_EXIT {
        if (!unrefImage(destinationImage)) {
            freeSafe(bits);
        }
    };

    pixman_image_t* sourceImage = pixman_image_create_bits(
        sourceFormat,
        surface.width, surface.height,
        (uint32_t*)surface.image,
        (int)surface.stride
    );

    if (!sourceImage) {
        return 0;
    }

    SCOPE_EXIT {
        if (!unrefImage(sourceImage)) {
            freeSafe(bits);
        }
    };

    // we should only care about premultiplying if:
    // -the source format is PIXMAN_x8r8g8b8 (indicating we are meant to use it with maskImage)
    // -the destination format has alpha (because otherwise the colours will be unaffected by alpha)
    // -the destination format has RGB channels (because otherwise the colour data will be thrown out anyway)
    // -the image format is alpha (so we aren't creating an alpha channel for an opaque image)
    // we don't care about if the surface has alpha here
    // the source format will be PIXMAN_x8r8g8b8 if it does/it matters
    bool unpremultiply = sourceFormat == PIXMAN_x8r8g8b8
        && PIXMAN_FORMAT_A(destinationFormat)
        && PIXMAN_FORMAT_COLOR(destinationFormat)
        && isAlpha;

    // premultiply, only if we'll undo it later, and if the original image wasn't already premultiplied
    pixman_image_t* maskImage = unpremultiply
        ? premultiplyMaskImage(surface, sourceImage)
        : sourceImage;

    SCOPE_EXIT {
        if (maskImage && maskImage != sourceImage) {
            if (!unrefImage(maskImage)) {
                freeSafe(bits);
            }
        }
    };

    if (!setTransform(maskImage, surface, width, height)) {
        return 0;
    }

    if (!pixman_image_set_filter(maskImage, PIXMAN_FILTER_BILINEAR, NULL, 0)) {
        return 0;
    }

    // setting the repeat mode to pad prevents some semi-transparent lines at the edge of the image
    // (because the image is interpreted as being in the middle of a transparent void of pixels otherwise)
    pixman_image_set_repeat(maskImage, PIXMAN_REPEAT_PAD);

    // the actual resize happens here
    pixman_image_composite(
        PIXMAN_OP_SRC,
        maskImage, NULL, destinationImage,
        0, 0, 0, 0, 0, 0,
        width, height
    );

    // as a final step we need to unpremultiply
    // as also convert down to 16-bit colour as necessary
    if (convert) {
        const size_t COLOR16_SIZE = sizeof(M4Image::Color16);

        if (!stride) {
            stride = (size_t)width * COLOR16_SIZE;
        }

        unsigned char* convertedBits = convertImage((M4Image::Color32*)bits, width, height, stride, unpremultiply);

        if (!convertedBits) {
            return 0;
        }

        // this is necessary so that, if an error occurs with unreffing the images
        // that we return zero, because bits will become set to zero
        freeSafe(bits);
        bits = convertedBits;
    } else {
        stride = bitsStride;

        if (unpremultiply) {
            unpremultiplyColors((M4Image::Color32*)bits, width, height, stride);
        }
    }

    bitsScopeExit.dismiss();
    return bits;
}

static const mango::image::Format &IMAGE_HEADER_FORMAT_RGBA = FORMAT_MAP.at(M4Image::COLOR_FORMAT::RGBA32);

namespace M4Image {
    unsigned char* load(
        const char* extension,
        const unsigned char* address,
        size_t size,
        int width,
        int height,
        size_t &stride,
        COLOR_FORMAT colorFormat
    ) {
        MAKE_SCOPE_EXIT(strideScopeExit) {
            stride = 0;
        };

        if (!extension) {
            return 0;
        }

        if (!address) {
            return 0;
        }

        if (!width || !height) {
            return 0;
        }

        mango::image::ImageDecoder imageDecoder(mango::ConstMemory(address, size), extension);

        if (!imageDecoder.isDecoder()) {
            return 0;
        }

        mango::image::ImageHeader imageHeader = imageDecoder.header();

        // don't care, not implemented
        if (imageHeader.premultiplied) {
            return 0;
        }

        bool resize = width != imageHeader.width || height != imageHeader.height;
        bool convert = colorFormat == COLOR_FORMAT::AL16;

        pixman_format_code_t sourceFormat = PIXMAN_x8r8g8b8;
        pixman_format_code_t destinationFormat = PIXMAN_a8r8g8b8;

        if (resize) {
            colorFormat = getResizeColorFormat(imageHeader.format == IMAGE_HEADER_FORMAT_RGBA, colorFormat, sourceFormat, destinationFormat);
        }

        mango::image::Surface surface = mango::image::Surface();

        try {
            surface.format = FORMAT_MAP.at(colorFormat);
            surface.width = imageHeader.width;
            surface.height = imageHeader.height;
            surface.stride = (stride && !resize) ? stride : (size_t)imageHeader.width * (size_t)surface.format.bytes();
            decodeSurfaceImage(surface, imageDecoder);
        } catch (...) {
            freeSafe(surface.image);
            return 0;
        }

        if (!surface.image) {
            return 0;
        }

        // if we don't need to resize the image (width and height matches) then job done
        if (!resize) {
            stride = surface.stride;
            strideScopeExit.dismiss();
            return surface.image;
        }
        
        unsigned char* bits = resizeImage(
            surface,
            width,
            height,
            stride,
            convert,
            imageHeader.format.isAlpha(),
            sourceFormat,
            destinationFormat
        );

        if (bits) {
            strideScopeExit.dismiss();
        }
        return bits;
    }

    M4IMAGE_API unsigned char* M4IMAGE_CALL save(
        const char* extension,
        const void* image,
        size_t &size,
        int width,
        int height,
        size_t stride,
        COLOR_FORMAT colorFormat,
        float quality
    ) {
        MAKE_SCOPE_EXIT(sizeScopeExit) {
            size = 0;
        };

        if (!extension) {
            return 0;
        }

        if (!image) {
            return 0;
        }

        if (!width || !height) {
            return 0;
        }

        unsigned char* bits = 0;

        try {
            const mango::image::Format &FORMAT = FORMAT_MAP.at(colorFormat);

            const mango::image::Surface SURFACE = mango::image::Surface(
                width, height,
                FORMAT, stride ? stride : (size_t)width * (size_t)FORMAT.bytes(),
                image
            );

            bits = encodeSurfaceImage(SURFACE, extension, size, quality);
        } catch (...) {
            return 0;
        }

        if (bits) {
            sizeScopeExit.dismiss();
        }
        return bits;
    }

    M4IMAGE_API unsigned char* M4IMAGE_CALL blit(
        const void* image,
        int inputWidth,
        int inputHeight,
        size_t inputStride,
        COLOR_FORMAT inputColorFormat,
        int outputWidth,
        int outputHeight,
        size_t &outputStride,
        COLOR_FORMAT outputColorFormat
    ) {
        MAKE_SCOPE_EXIT(outputStrideScopeExit) {
            outputStride = 0;
        };

        if (!image) {
            return 0;
        }

        if (!inputWidth || !inputHeight) {
            return 0;
        }

        if (!outputWidth || !outputHeight) {
            return 0;
        }

        bool resize = inputWidth != outputWidth || inputHeight != outputHeight;
        bool convert = outputColorFormat == COLOR_FORMAT::AL16;
        bool isAlpha = false;

        pixman_format_code_t sourceFormat = PIXMAN_x8r8g8b8;
        pixman_format_code_t destinationFormat = PIXMAN_a8r8g8b8;

        if (resize) {
            outputColorFormat = getResizeColorFormat(inputColorFormat == M4Image::COLOR_FORMAT::RGBA32, outputColorFormat, sourceFormat, destinationFormat);
        }

        mango::image::Surface outputSurface = mango::image::Surface();

        try {
            const mango::image::Format &INPUT_FORMAT = FORMAT_MAP.at(inputColorFormat);

            const mango::image::Surface INPUT_SURFACE = mango::image::Surface(
                inputWidth, inputHeight,
                INPUT_FORMAT, inputStride ? inputStride : (size_t)inputWidth * (size_t)INPUT_FORMAT.bytes(),
                image
            );

            isAlpha = INPUT_SURFACE.format.isAlpha();

            // the resize is not done here, so the input width and height is used for the output surface
            outputSurface.format = FORMAT_MAP.at(outputColorFormat);
            outputSurface.width = inputWidth;
            outputSurface.height = inputHeight;
            outputSurface.stride = (outputStride && !resize) ? outputStride : (size_t)outputWidth * (size_t)outputSurface.format.bytes();
            blitSurfaceImage(INPUT_SURFACE, outputSurface);
        } catch (...) {
            return 0;
        }

        if (!outputSurface.image) {
            return 0;
        }

        if (!resize) {
            outputStride = outputSurface.stride;
            outputStrideScopeExit.dismiss();
            return outputSurface.image;
        }

        unsigned char* bits = resizeImage(
            outputSurface,
            outputWidth,
            outputHeight,
            outputStride,
            convert,
            isAlpha,
            sourceFormat,
            destinationFormat
        );

        if (bits) {
            outputStrideScopeExit.dismiss();
        }
        return bits;
    }

    void* malloc(size_t size) {
        return mallocProc(size);
    }

    void free(void* block) {
        freeProc(block);
    }

    void* realloc(void* block, size_t size) {
        return reallocProc(block, size);
    }

    bool getInfo(
        const char* extension,
        const unsigned char* address,
        size_t size,
        int* widthPointer,
        int* heightPointer,
        uint32_t* bitsPointer,
        bool* alphaPointer
    ) {
        if (!extension) {
            return false;
        }

        if (!address) {
            return false;
        }

        mango::image::ImageDecoder imageDecoder(mango::ConstMemory(address, size), extension);

        if (!imageDecoder.isDecoder()) {
            return false;
        }

        mango::image::ImageHeader imageHeader = imageDecoder.header();

        if (widthPointer) {
            *widthPointer = imageHeader.width;
        }

        if (heightPointer) {
            *heightPointer = imageHeader.height;
        }

        if (bitsPointer) {
            *bitsPointer = imageHeader.format.bits;
        }

        if (alphaPointer) {
            *alphaPointer = imageHeader.format.isAlpha();
        }
        return true;
    }

    void setAllocator(MallocProc _mallocProc, FreeProc _freeProc, ReallocProc _reallocProc) {
        mallocProc = _mallocProc;
        freeProc = _freeProc;
        reallocProc = _reallocProc;
    }
};