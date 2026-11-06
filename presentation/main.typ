#import "@preview/minideck:0.2.1"
#let (template, slide, title-slide, pause, uncover, only) = minideck.config()
#show: template

#title-slide[
  = Slides with `minideck`
  == Some examples
  John Doe
  === rust-gpu setup
  - cargo install cargo-gpu

  - set up shader crate
    - defines data for sharing between main and GPU
  - set up caller crate
  - 
  - install vulkan https://vulkan.lunarg.com/sdk/home#mac?
export VULKAN_SDK=/Users/cristina/VulkanSDK/1.4.363.0
export MOLTENVK_ICD=/Users/cristina/VulkanSDK/1.4.363.0/macOS/share/vulkan/icd.d/MoltenVK_icd.json
  export MOLTENVK_LIB=/Users/cristina/VulkanSDK/1.4.363.0/macOS/lib
  #datetime.today().display()
]

#slide[
  == platforms
  - WebGPU (not actually for web, more like OpenGL)
  - Cuda
  - Vulkan
  - Metal
  - RocM HIP

]

#slide[
  = Render shaders
  == Render pipeline
  === vertex shaders transform vertices 
  === fragment shaders transform pixels

  = SDKs
  - RustGPU

]

#slide[
  = Compute shaders
  == memory in 
  == apply transformations
  == memory out
  = SDKs
  - RustGPU
  - cuda-rs
  - cubecl
]
