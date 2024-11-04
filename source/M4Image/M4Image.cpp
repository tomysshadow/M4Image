#include "M4Image/M4Image.h"
#include "M4Image/scope_guard.hpp"
#include <array>
#include <map>
#include <optional>
#include <memory>

#include <mango/image/surface.hpp>
#include <mango/image/decoder.hpp>
#include <mango/image/quantize.hpp>
#include <pixman.h>

// done to ensure constant initialization and avoid static initialization order fiasco
static constexpr M4Image::Allocator defaultAllocator;
M4Image::Allocator M4Image::allocator = defaultAllocator;

struct MallocDeleter {
    void operator()(void* block) {
        M4Image::allocator.freeSafe(block);
    }
};

typedef std::array<unsigned char, USHRT_MAX + 1> CHANNEL_UNPREMULTIPLIER_ARRAY;

constexpr CHANNEL_UNPREMULTIPLIER_ARRAY createChannelUnpremultiplierArray() {
    CHANNEL_UNPREMULTIPLIER_ARRAY channelUnpremultiplierArray = {};

    // note: the alpha, divided by two, is added to the channel
    // so the channel is scaled instead of stripped (it works out to rounding the number, instead of flooring)
    // alpha starts at one, since if it's zero the colour is invisible anyway (and thus would be a divide by zero)
    const size_t DIVIDE_BY_TWO = 1;

    for (int channel = 0; channel <= UCHAR_MAX; channel++) {
        for (int alpha = 1; alpha <= UCHAR_MAX; alpha++) {
            channelUnpremultiplierArray[(channel << CHAR_BIT) | alpha] = clampUCHAR(((channel * UCHAR_MAX) + (alpha >> DIVIDE_BY_TWO)) / alpha);
        }
    }
    return channelUnpremultiplierArray;
}

// aligned to nearest 64 bytes so it is on cache lines
// note: there is a specific IntelliSense error that it only shows for a std::array of size 65536 bytes
// this is an IntelliSense bug, it compiles correctly: https://github.com/microsoft/vscode-cpptools/issues/5833
alignas(64) static constexpr CHANNEL_UNPREMULTIPLIER_ARRAY UNPREMULTIPLIER_ARRAY = createChannelUnpremultiplierArray();

#define UNPREMULTIPLY_CHANNEL(channel, alpha) (UNPREMULTIPLIER_ARRAY[((channel) << CHAR_BIT) | (alpha)])

void convertColors(
    M4Image::Color32* colorPointer,
    size_t width,
    size_t height,
    size_t stride,
    unsigned char* imagePointer,
    bool unpremultiply
) {
    const size_t CHANNEL_LUMINANCE = 2;
    const size_t CHANNEL_ALPHA = 3;

    M4Image::Color16* luminancePointer = (M4Image::Color16*)imagePointer;

    if (unpremultiply) {
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

            imagePointer += stride;
            luminancePointer = (M4Image::Color16*)imagePointer;
        }
    } else {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                *luminancePointer++ = *(M4Image::Color16*)&colorPointer++->channels[CHANNEL_LUMINANCE];
            }

            imagePointer += stride;
            luminancePointer = (M4Image::Color16*)imagePointer;
        }
    }
}

void unpremultiplyColors(
    M4Image::Color32* colorPointer,
    size_t width,
    size_t height,
    size_t stride
) {
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

/*
void grayscaleColors(
    M4Image::Color32* colorPointer,
    size_t width,
    size_t height,
    size_t stride,
    bool linear = false
) {
    const size_t CHANNEL_R = 0;
    const size_t CHANNEL_G = 1;
    const size_t CHANNEL_B = 2;
    const size_t CHANNEL_L = 2;

    unsigned char* rowPointer = (unsigned char*)colorPointer;

    if (linear) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                unsigned char &r = colorPointer->channels[CHANNEL_R];
                unsigned char &g = colorPointer->channels[CHANNEL_G];
                unsigned char &b = colorPointer->channels[CHANNEL_B];

                // fast grayscale approximation
                // https://stackoverflow.com/a/596241/3591734
                colorPointer++->channels[CHANNEL_L] = ((r << 1) + r + (g << 2) + b) >> 3;
            }

            rowPointer += stride;
            colorPointer = (M4Image::Color32*)rowPointer;
        }
        return;
    }

    unsigned short rLinear = 0;
    unsigned short gLinear = 0;
    unsigned short bLinear = 0;

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            unsigned char &r = colorPointer->channels[CHANNEL_R];
            unsigned char &g = colorPointer->channels[CHANNEL_G];
            unsigned char &b = colorPointer->channels[CHANNEL_B];

            // fast sRGB to linear approximation
            // https://gamedev.stackexchange.com/a/105045/54039
            rLinear = r * r;
            gLinear = g * g;
            bLinear = b * b;

            colorPointer++->channels[CHANNEL_L] = sqrt(((rLinear << 1) + rLinear + (gLinear << 2) + bLinear) >> 3);
        }

        rowPointer += stride;
        colorPointer = (M4Image::Color32*)rowPointer;
    }
}
*/

// the mango::core::MemoryStream class does not allow using a custom allocator
// so we must implement our own
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
    M4Image::allocator.freeSafe(state.data);
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

    M4Image::allocator.reAllocSafe(state.data, capacity);
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
    {M4Image::COLOR_FORMAT::LA16, mango::image::LuminanceFormat(16, 0x000000FF, 0x0000FF00)},
    {M4Image::COLOR_FORMAT::A8, mango::image::Format(8, mango::image::Format::UNORM, mango::image::Format::A, 8, 0, 0, 0)},
    {M4Image::COLOR_FORMAT::L8, mango::image::LuminanceFormat(8, 0x000000FF, 0x00000000)},
    {M4Image::COLOR_FORMAT::XXXL32, mango::image::LuminanceFormat(32, 0xFF000000, 0x00000000)},
    {M4Image::COLOR_FORMAT::XXLA32, mango::image::LuminanceFormat(32, 0x00FF0000, 0xFF000000)},
};

static const mango::image::Format &IMAGE_HEADER_FORMAT_RGBA = FORMAT_MAP.at(M4Image::COLOR_FORMAT::RGBA32);

void blitSurfaceImage(
    const mango::image::Surface &inputSurface,
    mango::image::Surface &outputSurface,
    bool linear = false
) {
    std::optional<mango::image::LuminanceBitmap> luminanceBitmapOptional = std::nullopt;

    if (!inputSurface.format.isLuminance() && outputSurface.format.isLuminance()) {
        luminanceBitmapOptional.emplace(inputSurface, outputSurface.format.isAlpha(), linear);
    }

    const mango::image::Surface &SOURCE_SURFACE = luminanceBitmapOptional.has_value() ? luminanceBitmapOptional.value() : inputSurface;

    // if we can avoid a blit and do a direct memory copy, do that instead
    // (it is assumed the caller has ensured the width/height match)
    if (SOURCE_SURFACE.format == outputSurface.format && SOURCE_SURFACE.stride == outputSurface.stride) {
        // if we're direct and the image pointers match, they are already equal so copying is unnecessary
        if (SOURCE_SURFACE.image == outputSurface.image) {
            return;
        }

        if (memcpy_s(outputSurface.image, outputSurface.stride * (size_t)outputSurface.height, SOURCE_SURFACE.image, SOURCE_SURFACE.stride * (size_t)SOURCE_SURFACE.height)) {
            throw std::runtime_error("Failed to Copy Memory");
        }
    } else {
        outputSurface.blit(0, 0, SOURCE_SURFACE);
    }
}

void decodeSurfaceImage(
    const mango::image::Surface &luminanceSurface,
    mango::image::Surface &surface,
    mango::image::ImageDecoder &imageDecoder,
    bool linear = false
) {
    // uncomment the second argument to disable multithreading for testing purposes
    mango::image::ImageDecodeStatus status = imageDecoder.decode(luminanceSurface/*, {nullptr, true, false}*/);

    // status is false if decoding the image failed
    if (!status) {
        MANGO_EXCEPTION("[INFO] {}", status.info);
    }

    // for grayscale images we may need to blit them to a luminance format
    blitSurfaceImage(luminanceSurface, surface, linear);
}

unsigned char* encodeSurfaceImage(
    const mango::image::Surface &surface,
    const char* extension,
    size_t &size,
    float quality = 0.90f
) {
    MAKE_SCOPE_EXIT(sizeScopeExit) {
        size = 0;
    };

    AllocatorStream allocatorStream;
    mango::image::ImageEncodeStatus status = surface.save(allocatorStream, extension, { {}, {}, quality });

    if (!status) {
        MANGO_EXCEPTION("[INFO] {}", status.info);
    }

    size = allocatorStream.size();
    unsigned char* bits = allocatorStream.acquire();

    // can technically happen, but probably never will
    if (!bits) {
        throw std::bad_alloc();
    }

    sizeScopeExit.dismiss();
    return bits;
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
    M4Image::COLOR_FORMAT colorFormat,
    pixman_format_code_t &sourceFormat,
    pixman_format_code_t &destinationFormat,
    bool rgba
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
        case M4Image::COLOR_FORMAT::LA16:
        case M4Image::COLOR_FORMAT::XXLA32:
        // for COLOR_FORMAT::LA16, mango reads the image in XXLA format
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
        throw std::bad_alloc();
    }

    MAKE_SCOPE_EXIT(maskImageScopeExit) {
        if (!unrefImage(maskImage)) {
            throw std::runtime_error("Failed to Unref Image");
        }
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
void setTransform(pixman_image_t* maskImage, const mango::image::Surface &surface, int width, int height) {
    // this is initialized by pixman_transform_init_identity
    pixman_transform_t transform;
    pixman_transform_init_identity(&transform);

    double sx = (double)surface.width / width;
    double sy = (double)surface.height / height;

    if (!pixman_transform_scale(&transform, NULL, pixman_double_to_fixed(sx), pixman_double_to_fixed(sy))) {
        throw std::runtime_error("Failed to Scale Transform");
    }

    if (!pixman_image_set_transform(maskImage, &transform)) {
        throw std::runtime_error("Failed to Set Image Transform");
    }
}

void resizeImage(
    mango::image::Surface &surface,
    pixman_format_code_t sourceFormat,
    pixman_format_code_t destinationFormat,
    int width,
    int height,
    size_t stride,
    unsigned char* imagePointer,
    bool convert,
    bool premultiplied
) {
    // we have to do a bunch of boilerplate setup stuff for the resize operation
    // bitsPointer MUST be a seperate buffer to surface.image (yes, even for upscaling)
    // if we aren't converting, we expect to get the destination image in the user requested stride
    // if we are converting, we expect it to be based on the format, for convenience's sake
    size_t bitsStride = stride;
    unsigned char* bitsPointer = imagePointer;

    typedef std::unique_ptr<unsigned char[], MallocDeleter> CONVERT_BITS_POINTER;
    CONVERT_BITS_POINTER convertBitsPointer = 0;

    if (convert) {
        const size_t BYTES = 3;

        bitsStride = (PIXMAN_FORMAT_BPP(destinationFormat) >> BYTES) * (size_t)width;

        convertBitsPointer = CONVERT_BITS_POINTER((unsigned char*)M4Image::allocator.mallocSafe(bitsStride * (size_t)height));
        bitsPointer = convertBitsPointer.get();
    }

    // create the destination image in the user's desired format
    // (unless we need to convert to 16-bit after, then we still make it 32-bit)
    pixman_image_t* destinationImage = pixman_image_create_bits(
        destinationFormat,
        width, height,
        (uint32_t*)bitsPointer,
        (int)bitsStride
    );

    if (!destinationImage) {
        throw std::bad_alloc();
    }

    SCOPE_EXIT {
        if (!unrefImage(destinationImage)) {
            throw std::runtime_error("Failed to Unref Image");
        }
    };

    pixman_image_t* sourceImage = pixman_image_create_bits(
        sourceFormat,
        surface.width, surface.height,
        (uint32_t*)surface.image,
        (int)surface.stride
    );

    if (!sourceImage) {
        throw std::bad_alloc();
    }

    SCOPE_EXIT {
        if (!unrefImage(sourceImage)) {
            throw std::runtime_error("Failed to Unref Image");
        }
    };

    // we should only care about premultiplying if:
    // -the image format isn't already premultiplied (then it's the caller's problem)
    // -the source format is PIXMAN_x8r8g8b8 (indicating we are meant to use it with maskImage)
    // -the destination format has alpha (because otherwise the colours will be unaffected by alpha)
    // -the destination format has RGB channels (because otherwise the colour data will be thrown out anyway)
    // we don't care about if the surface has alpha here
    // the source format will be PIXMAN_x8r8g8b8 if it does/it matters
    bool unpremultiply = !premultiplied
        && sourceFormat == PIXMAN_x8r8g8b8
        && PIXMAN_FORMAT_A(destinationFormat)
        && PIXMAN_FORMAT_COLOR(destinationFormat);

    // premultiply, only if we'll undo it later, and if the original image wasn't already premultiplied
    pixman_image_t* maskImage = unpremultiply
        ? premultiplyMaskImage(surface, sourceImage)
        : sourceImage;

    SCOPE_EXIT {
        if (maskImage != sourceImage) {
            if (!unrefImage(maskImage)) {
                throw std::runtime_error("Failed to Unref Image");
            }
        }
    };

    setTransform(maskImage, surface, width, height);

    if (!pixman_image_set_filter(maskImage, PIXMAN_FILTER_BILINEAR, NULL, 0)) {
        throw std::runtime_error("Failed to Set Filter");
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
        convertColors((M4Image::Color32*)bitsPointer, width, height, stride, imagePointer, unpremultiply);
        return;
    } else if (unpremultiply) {
        unpremultiplyColors((M4Image::Color32*)bitsPointer, width, height, stride);
    }
}

void M4Image::getInfo(
    const unsigned char* pointer,
    size_t size,
    const char* extension,
    bool* isAlphaPointer,
    uint32_t* bitsPointer,
    int* widthPointer,
    int* heightPointer,
    bool* linearPointer,
    bool* premultipliedPointer
) {
    if (!pointer) {
        throw std::invalid_argument("pointer must not be zero");
    }

    if (!extension) {
        extension = ".";
    }

    mango::image::ImageDecoder imageDecoder(mango::ConstMemory(pointer, size), extension);

    if (!imageDecoder.isDecoder()) {
        throw std::logic_error("No Decoder");
    }

    mango::image::ImageHeader imageHeader = imageDecoder.header();

    if (isAlphaPointer) {
        *isAlphaPointer = imageHeader.format.isAlpha();
    }

    if (bitsPointer) {
        *bitsPointer = imageHeader.palette ? CHAR_BIT : imageHeader.format.bits;
    }

    if (widthPointer) {
        *widthPointer = imageHeader.width;
    }

    if (heightPointer) {
        *heightPointer = imageHeader.height;
    }

    if (linearPointer) {
        *linearPointer = imageHeader.linear;
    }

    if (premultipliedPointer) {
        *premultipliedPointer = imageHeader.premultiplied;
    }
}

M4Image::M4Image(int width, int height, size_t &stride, COLOR_FORMAT colorFormat, unsigned char* imagePointer) {
    create(width, height, stride, colorFormat, imagePointer);
}

M4Image::M4Image(int width, int height) {
    size_t stride = 0;
    create(width, height, stride);
}

M4Image::~M4Image() {
    destroy();
}

void M4Image::blit(const M4Image &m4Image, bool linear, bool premultiplied) {
    if (!imagePointer || !m4Image.imagePointer) {
        throw Invalid();
    }

    const mango::image::Surface INPUT_SURFACE(
        m4Image.width, m4Image.height,
        FORMAT_MAP.at(m4Image.colorFormat), m4Image.stride,
        m4Image.imagePointer
    );

    bool resize = width != m4Image.width || height != m4Image.height;

    pixman_format_code_t sourceFormat = PIXMAN_x8r8g8b8;
    pixman_format_code_t destinationFormat = PIXMAN_a8r8g8b8;

    const mango::image::Format &OUTPUT_FORMAT = FORMAT_MAP.at(
        resize

        ? getResizeColorFormat(
            colorFormat,
            sourceFormat,
            destinationFormat,
            m4Image.colorFormat == M4Image::COLOR_FORMAT::RGBA32
        )

        : colorFormat
    );

    size_t outputSurfaceStride = stride;
    std::unique_ptr<mango::u8[], MallocDeleter> outputSurfaceImagePointer = nullptr;

    if (resize) {
        outputSurfaceStride = (size_t)m4Image.width * (size_t)OUTPUT_FORMAT.bytes();

        outputSurfaceImagePointer = std::unique_ptr<mango::u8[], MallocDeleter>(
            (mango::u8*)M4Image::allocator.mallocSafe(outputSurfaceStride * (size_t)m4Image.height)
        );
    }

    // the resize is not done here, so the input width and height is used for the output surface
    mango::image::Surface outputSurface(
        m4Image.width, m4Image.height,
        OUTPUT_FORMAT, outputSurfaceStride,
        outputSurfaceImagePointer ? outputSurfaceImagePointer.get() : imagePointer
    );

    try {
        blitSurfaceImage(INPUT_SURFACE, outputSurface, linear);
    } catch (mango::Exception) {
        throw std::runtime_error("Failed to Blit Surface Image");
    }

    if (!resize) {
        return;
    }

    // for our purposes, if the image is opaque, it is as if the image were premultiplied
    resizeImage(
        outputSurface,
        sourceFormat,
        destinationFormat,
        width,
        height,
        stride,
        imagePointer,
        colorFormat == COLOR_FORMAT::LA16,
        premultiplied || !INPUT_SURFACE.format.isAlpha()
    );
}

void M4Image::load(const unsigned char* pointer, size_t size, const char* extension, bool &linear, bool &premultiplied) {
    MAKE_SCOPE_EXIT(linearScopeExit) {
        linear = false;
    };

    MAKE_SCOPE_EXIT(premultipliedScopeExit) {
        premultiplied = false;
    };

    if (!imagePointer) {
        throw Invalid();
    }

    if (!pointer) {
        throw std::invalid_argument("pointer must not be zero");
    }

    if (!extension) {
        extension = ".";
    }

    mango::image::ImageDecoder imageDecoder(mango::ConstMemory(pointer, size), extension);

    if (!imageDecoder.isDecoder()) {
        throw std::logic_error("No Decoder");
    }

    mango::image::ImageHeader imageHeader = imageDecoder.header();

    bool resize = width != imageHeader.width || height != imageHeader.height;
    linear = imageHeader.linear;
    premultiplied = imageHeader.premultiplied;

    pixman_format_code_t sourceFormat = PIXMAN_x8r8g8b8;
    pixman_format_code_t destinationFormat = PIXMAN_a8r8g8b8;

    const mango::image::Format &SURFACE_FORMAT = FORMAT_MAP.at(
        resize
        
        ? getResizeColorFormat(
            colorFormat,
            sourceFormat,
            destinationFormat,
            imageHeader.format == IMAGE_HEADER_FORMAT_RGBA
        )
            
        : colorFormat
    );

    size_t surfaceStride = stride;
    std::unique_ptr<mango::u8[], MallocDeleter> surfaceImagePointer = nullptr;

    if (resize) {
        surfaceStride = (size_t)imageHeader.width * (size_t)SURFACE_FORMAT.bytes();
        surfaceImagePointer = std::unique_ptr<mango::u8[], MallocDeleter>((mango::u8*)M4Image::allocator.mallocSafe(surfaceStride * (size_t)imageHeader.height));
    }

    mango::image::Surface surface(
        imageHeader.width, imageHeader.height,
        SURFACE_FORMAT, surfaceStride,
        surfaceImagePointer ? surfaceImagePointer.get() : imagePointer
    );

    // scope for temporary luminance surface stuff
    {
        bool isLuminance = SURFACE_FORMAT.isLuminance();

        // LuminanceBitmap uses RGBA natively, so import to that if the blit format is luminance
        const mango::image::Format &LUMINANCE_SURFACE_FORMAT = isLuminance
            ? IMAGE_HEADER_FORMAT_RGBA
            : SURFACE_FORMAT;

        size_t luminanceSurfaceStride = surface.stride;
        std::unique_ptr<mango::u8[], MallocDeleter> luminanceSurfaceImagePointer = nullptr;

        if (isLuminance) {
            luminanceSurfaceStride = (size_t)imageHeader.width * (size_t)LUMINANCE_SURFACE_FORMAT.bytes();

            // if we are dealing with luminance
            // there will always be a LuminanceBitmap intermediary
            // so we can safely use the same memory for
            // both the RGBA32 and XXLA32 side of things, as we'll never
            // end up interpreting the same memory as a different format for the blit
            // we should only allocate this if it needs to be a different size
            // like RGBA32 to L8 for example
            if (luminanceSurfaceStride != surfaceStride) {
                luminanceSurfaceImagePointer = std::unique_ptr<mango::u8[], MallocDeleter>((mango::u8*)M4Image::allocator.mallocSafe(luminanceSurfaceStride * (size_t)imageHeader.height));
            }
        }

        const mango::image::Surface LUMINANCE_SURFACE(
            imageHeader.width, imageHeader.height,
            LUMINANCE_SURFACE_FORMAT, luminanceSurfaceStride,
            luminanceSurfaceImagePointer ? luminanceSurfaceImagePointer.get() : surface.image
        );

        try {
            decodeSurfaceImage(
                LUMINANCE_SURFACE,
                surface,
                imageDecoder,
                linear
            );
        } catch (mango::Exception) {
            throw std::runtime_error("Failed to Decode Surface Image");
        }
    }

    // if we don't need to resize the image (width and height matches) then job done
    if (!resize) {
        premultipliedScopeExit.dismiss();
        linearScopeExit.dismiss();
        return;
    }
        
    // here we use the same trick where if the image is opaque, we say it's premultiplied
    // however the caller should not get to know this
    resizeImage(
        surface,
        sourceFormat,
        destinationFormat,
        width,
        height,
        stride,
        imagePointer,
        colorFormat == COLOR_FORMAT::LA16,
        premultiplied || !imageHeader.format.isAlpha()
    );

    premultipliedScopeExit.dismiss();
    linearScopeExit.dismiss();
}

void M4Image::load(const unsigned char* pointer, size_t size, const char* extension, bool &linear) {
    bool premultiplied = false;
    load(pointer, size, extension, linear, premultiplied);
}

void M4Image::load(const unsigned char* pointer, size_t size, const char* extension) {
    bool linear = false;
    load(pointer, size, extension, linear);
}

unsigned char* M4Image::save(size_t &size, const char* extension, float quality) const {
    MAKE_SCOPE_EXIT(sizeScopeExit) {
        size = 0;
    };

    if (!imagePointer) {
        throw Invalid();
    }

    if (!extension) {
        extension = ".";
    }

    const mango::image::Surface SURFACE(
        width, height,
        FORMAT_MAP.at(colorFormat), stride,
        imagePointer
    );

    unsigned char* bits = 0;

    try {
        bits = encodeSurfaceImage(SURFACE, extension, size, quality);
    } catch (mango::Exception) {
        throw std::runtime_error("Failed to Encode Surface Image");
    }

    sizeScopeExit.dismiss();
    return bits;
}

unsigned char* M4Image::acquire() {
    unsigned char* imagePointer = this->imagePointer;
    this->imagePointer = 0;
    return imagePointer;
}

void M4Image::create(int width, int height, size_t &stride, COLOR_FORMAT colorFormat, unsigned char* imagePointer) {
    if (!width || !height) {
        throw std::invalid_argument("width and height must not be zero");
    }

    if (!stride) {
        stride = (size_t)width * (size_t)FORMAT_MAP.at(colorFormat).bytes();
    }

    if (!imagePointer) {
        imagePointer = (unsigned char*)M4Image::allocator.mallocSafe(stride * (size_t)height);
        owner = true;
    }

    this->width = width;
    this->height = height;
    this->stride = stride;
    this->colorFormat = colorFormat;
    this->imagePointer = imagePointer;
}

void M4Image::destroy() {
    if (owner) {
        M4Image::allocator.freeSafe(imagePointer);
    }
}