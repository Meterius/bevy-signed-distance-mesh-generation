use cudarc::driver::DeviceRepr;

pub mod cuda;

unsafe impl DeviceRepr for cuda::GlobalsBuffer {}
unsafe impl DeviceRepr for cuda::CameraBuffer {}
unsafe impl DeviceRepr for cuda::Rgba {}
unsafe impl DeviceRepr for cuda::RenderTexture {}
unsafe impl DeviceRepr for cuda::Point {}
unsafe impl DeviceRepr for cuda::NaiveTriMesh {}
unsafe impl DeviceRepr for cuda::Vertex {}
unsafe impl DeviceRepr for cuda::BlockPartition {}
