# M4Image
## By Anthony Kleine

M4Image performs fast image blitting, loading, and saving for [Myst IV: Revolution.](https://github.com/tomysshadow/M4Revolution/) It is only intended for use by the Myst IV: Revolution project: its API is subject to change at any time in order to meet the needs of said project, and basic features that would normally be expected of image libraries (such as the ability to load from a file instead of memory) will not be added so long as they are not needed for said project. The only reason this is seperated into its own repo is to prevent a circular dependency with [libzap.](https://github.com/HumanGamer/libzap) With that said, feel free to fork it if you'd like to adapt it for something else!

M4Image is really just a wrapper around [mango](https://github.com/t0rakka/mango) and [Pixman](https://www.pixman.org), which you should probably use instead for your own projects.

# Dependencies

M4Image depends on these libraries.

- [scope_guard](https://github.com/Neargye/scope_guard) by Neargye
- [mango](https://github.com/t0rakka/mango) by t0rakka
- [Pixman](https://www.pixman.org) by freedesktop.org