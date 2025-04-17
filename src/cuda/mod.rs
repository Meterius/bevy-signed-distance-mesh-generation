use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use bevy::log::{error, info};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::driver::sys::CUdeviceptr;
use cudarc::nvrtc::Ptx;
use crate::bindings::cuda::{BLOCK_SIZE, MESH_GENERATION_BB_SIZE, MESH_GENERATION_INIT_FACTOR};

struct DynamicCudaSlice<T> {
    device: Arc<CudaDevice>,
    data: Option<CudaSlice<T>>,
}

impl<T: DeviceRepr> DynamicCudaSlice<T> {
    unsafe fn get_or_alloc_sync(&mut self, length: usize) -> CUdeviceptr {
        if self.data.as_ref().is_none_or(|data| data.len() < length) {
            self.data = Some(self.device.alloc(length).unwrap());
        }

        return std::mem::transmute(*(&self.data.as_ref().unwrap()).device_ptr());
    }
}

pub struct CudaHandler {
    device: Arc<CudaDevice>,

    func_compute_render: CudaFunction,

    func_compute_refine_voxel_field_by_sdf: CudaFunction,
    func_compute_surface_triangles_from_voxel_field_by_sdf: CudaFunction,

    render_texture_buffer: DynamicCudaSlice<crate::bindings::cuda::Rgba>,
    voxel_field_input_buffer: DynamicCudaSlice<crate::bindings::cuda::Point>,
    voxel_field_output_buffer: DynamicCudaSlice<crate::bindings::cuda::Point>,
    mesh_triangles_buffer: DynamicCudaSlice<crate::bindings::cuda::Triangle>,

    _marker: PhantomData<bool>
}

pub struct CudaVoxelField {
    pub voxels: Vec<crate::bindings::cuda::Point>,
    pub voxel_size: crate::bindings::cuda::Point,

    _marker: PhantomData<bool>
}

impl CudaHandler {
    pub fn new() -> CudaHandler {
        let start = std::time::Instant::now();

        let device = CudaDevice::new(0).unwrap();

        info!("CUDA Device Creation took {:.2?} seconds", start.elapsed());

        let start = std::time::Instant::now();

        device
            .load_ptx(
                Ptx::from_src(include_str!(
                    "../../assets/cuda/compiled/compute_render.ptx"
                )),
                "compute_render",
                &["compute_render"],
            )
            .unwrap();

        device
            .load_ptx(
                Ptx::from_src(include_str!(
                    "../../assets/cuda/compiled/compute_mesh_generation.ptx"
                )),
                "compute_mesh_generation",
                &[
                    "compute_refine_voxel_field_by_sdf",
                    "compute_surface_triangles_from_voxel_field_by_sdf",
                ],
            )
            .unwrap();

        let func_compute_render = device.get_func("compute_render", "compute_render").unwrap();
        let func_compute_surface_triangles_from_voxel_field_by_sdf = device
            .get_func("compute_mesh_generation", "compute_surface_triangles_from_voxel_field_by_sdf")
            .unwrap();
        let func_compute_refine_voxel_field_by_sdf = device
            .get_func(
                "compute_mesh_generation",
                "compute_refine_voxel_field_by_sdf",
            )
            .unwrap();

        info!("CUDA PTX Loading took {:.2?} seconds", start.elapsed());

        let render_texture_buffer = DynamicCudaSlice { device: device.clone(), data: None };
        let voxel_field_input_buffer = DynamicCudaSlice { device: device.clone(), data: None };
        let voxel_field_output_buffer = DynamicCudaSlice { device: device.clone(), data: None };
        let mesh_triangles_buffer = DynamicCudaSlice { device: device.clone(), data: None };

        return Self {
            device, func_compute_render, func_compute_surface_triangles_from_voxel_field_by_sdf, func_compute_refine_voxel_field_by_sdf,
            mesh_triangles_buffer, voxel_field_input_buffer, voxel_field_output_buffer, render_texture_buffer, _marker: PhantomData {},
        }
    }

    pub fn create_cuda_voxel_field() -> CudaVoxelField {
        const SIZE: f32 = MESH_GENERATION_BB_SIZE as f32 / MESH_GENERATION_INIT_FACTOR as f32;

        return CudaVoxelField {
            voxel_size: crate::bindings::cuda::Point { x: SIZE, y: SIZE, z: SIZE },
            voxels: (0..MESH_GENERATION_INIT_FACTOR as usize)
                .flat_map(move |x| {
                    (0..MESH_GENERATION_INIT_FACTOR as usize).flat_map(move |y| {
                        (0..MESH_GENERATION_INIT_FACTOR as usize).map(move |z| crate::bindings::cuda::Point {
                            x: (x as f32) * SIZE - (MESH_GENERATION_BB_SIZE as f32) / 2.0f32,
                            y: (y as f32) * SIZE - (MESH_GENERATION_BB_SIZE as f32) / 2.0f32,
                            z: (z as f32) * SIZE - (MESH_GENERATION_BB_SIZE as f32) / 2.0f32,
                        })
                    })
                }).collect(),
            _marker: PhantomData {},
        };
    }

    pub fn refine_voxel_field(&mut self, field: &mut CudaVoxelField) {
        let upper_output_voxel_count = field.voxels.len() * 8;
        let ouput_voxel_size = crate::bindings::cuda::Point {
            x: field.voxel_size.x / 2.0f32,
            y: field.voxel_size.y / 2.0f32,
            z: field.voxel_size.z / 2.0f32,
        };

        info!(
            "Refining Voxel Field; Current Voxel Count: {}; Current Voxel Size: {:?}; Worst Case Voxel Count: {};",
            field.voxels.len(), field.voxel_size, upper_output_voxel_count
        );

        if field.voxels.len() != 0 {
            let input_ptr = unsafe { self.voxel_field_input_buffer.get_or_alloc_sync(field.voxels.len()) };
            let output_ptr = unsafe { self.voxel_field_output_buffer.get_or_alloc_sync(upper_output_voxel_count) };

            unsafe {
                cudarc::driver::result::memcpy_htod_sync(
                    input_ptr,
                    field.voxels.as_slice(),
                )
                    .unwrap()
            };
            
            unsafe {
                self
                    .func_compute_refine_voxel_field_by_sdf
                    .clone()
                    .launch(
                        LaunchConfig {
                            grid_dim: (
                                (field.voxels.len() as f32 / BLOCK_SIZE as f32).ceil() as u32,
                                1,
                                1,
                            ),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            crate::bindings::cuda::VoxelField {
                                voxel_size: field.voxel_size,
                                voxel_count: field.voxels.len() as _,
                                voxels: std::mem::transmute(input_ptr),
                            },
                            crate::bindings::cuda::VoxelField {
                                voxel_size: crate::bindings::cuda::Point { x: 0.0f32, y: 0.0f32, z: 0.0f32 },
                                voxel_count: upper_output_voxel_count as _,
                                voxels: std::mem::transmute(output_ptr),
                            },
                        ),
                    )
                    .unwrap()
            };

            field.voxels.resize(
                upper_output_voxel_count,
                crate::bindings::cuda::Point::default(),
            );

            unsafe {
                cudarc::driver::result::memcpy_dtoh_sync(
                    field.voxels.as_mut_slice(),
                    output_ptr
                )
                    .unwrap()
            };

            field.voxels
                .retain(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
            field.voxel_size = ouput_voxel_size;
        }

        info!(
            "Refining Voxel Field; Voxel Count: {}; Voxel Size: {:?}",
            field.voxels.len(),
            field.voxel_size,
        );
    }

    pub fn voxel_field_to_mesh(&mut self, field: &CudaVoxelField) -> obj::ObjData {
        let upper_triangle_count = field.voxels.len() * 5;

        info!(
            "Voxel Field Mesh Generation; Current Voxel Count: {}; Current Voxel Size: {:?}; Worst Case Triangle Count: {};",
            field.voxels.len(),
            field.voxel_size,
            upper_triangle_count
        );

        let input_ptr = unsafe { self.voxel_field_input_buffer.get_or_alloc_sync(field.voxels.len()) };
        let triangles_ptr = unsafe { self.mesh_triangles_buffer.get_or_alloc_sync(upper_triangle_count) };

        if field.voxels.len() != 0 {
            unsafe {
                cudarc::driver::result::memcpy_htod_sync(
                    input_ptr,
                    field.voxels.as_slice(),
                )
                    .unwrap()
            };
            
            unsafe {
                self
                    .func_compute_surface_triangles_from_voxel_field_by_sdf
                    .clone()
                    .launch(
                        LaunchConfig {
                            grid_dim: (
                                (field.voxels.len() as f32 / BLOCK_SIZE as f32).ceil() as u32,
                                1,
                                1,
                            ),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            crate::bindings::cuda::VoxelField {
                                voxels: std::mem::transmute(input_ptr),
                                voxel_count: field.voxels.len() as _,
                                voxel_size: field.voxel_size,
                            },
                            triangles_ptr,
                        ),
                    )
                    .unwrap()
            };

            let mut raw_triangles = vec![crate::bindings::cuda::Triangle::default(); upper_triangle_count];

            unsafe {
                cudarc::driver::result::memcpy_dtoh_sync(
                    raw_triangles.as_mut_slice(),
                    triangles_ptr,
                ).unwrap()
            };

            self.device.synchronize().unwrap();

            let mut indices = vec![];

            let mut vertices = vec![];
            let mut normals = vec![];

            let mut vertex_index_map = HashMap::<[i64; 3], usize>::default();

            let f32_discrete = |x: f32| (x * 10e4f32).round() as i64;

            let map_vertex_pos = |v: crate::bindings::cuda::Point| {
                [f32_discrete(v.x), f32_discrete(v.y), f32_discrete(v.z)]
            };

            let mut add_or_get_vertex = |v: crate::bindings::cuda::Vertex| {
                vertex_index_map
                    .entry(map_vertex_pos(v.position))
                    .or_insert_with(|| {
                        let new_index = vertices.len();
                        vertices.push([v.position.x, v.position.y, v.position.z]);
                        normals.push([v.normal.x, v.normal.y, v.normal.z]);
                        new_index
                    })
                    .clone()
            };

            for triangle in raw_triangles.into_iter() {
                if triangle.vertices[0].position.x.is_finite() {
                    indices.push([
                        add_or_get_vertex(triangle.vertices[0]),
                        add_or_get_vertex(triangle.vertices[1]),
                        add_or_get_vertex(triangle.vertices[2]),
                    ]);
                }
            }

            let vertex_count = vertices.len();
            let triangle_count = indices.len() / 3;

            info!("Voxel Field Mesh Generation; Vertex Count: {vertex_count}; Triangle Count: {triangle_count};");

            return obj::ObjData {
                position: vertices,
                normal: normals,
                texture: vec![[0.0, 0.0]],
                material_libs: Vec::new(),
                objects: vec![obj::Object {
                    name: String::from("default"),
                    groups: vec![obj::Group {
                        name: String::from("default"),
                        index: 0,
                        material: None,
                        polys: indices
                            .into_iter()
                            .map(|t| {
                                obj::SimplePolygon(
                                    t.into_iter()
                                        .map(|idx| obj::IndexTuple(idx, Some(0), Some(idx)))
                                        .collect(),
                                )
                            })
                            .collect(),
                    }],
                }],
            };
        } else {
            info!("Voxel Field Mesh Generation; Empty Mesh;");
            
            return obj::ObjData {
                position: Vec::new(),
                normal: Vec::new(),
                texture: vec![[0.0, 0.0]],
                material_libs: Vec::new(),
                objects: vec![obj::Object {
                    name: String::from("default"),
                    groups: vec![obj::Group {
                        name: String::from("default"),
                        index: 0,
                        material: None,
                        polys: Vec::new(),
                    }],
                }],
            };
        }
    }

    pub fn render(
        &mut self,
        target_image: &mut [u8],
        globals: crate::bindings::cuda::GlobalsBuffer,
        camera: crate::bindings::cuda::CameraBuffer
    ) {
        let range_id = nvtx::range_start!("Render System Wait For Previous Frame");

        let render_width = globals.render_texture_size[0] as usize;
        let render_height = globals.render_texture_size[1] as usize;
        let render_buffer_size = 4 * render_width * render_height;

        if target_image.len() != render_buffer_size {
            error!("Target image has size {render_buffer_size}, but should have size {render_buffer_size}.");
        }

        let render_texture_buffer_ptr = unsafe {
            self.render_texture_buffer.get_or_alloc_sync(render_buffer_size)
        };

        nvtx::range_end!(range_id);

        let range_id = nvtx::range_start!("Render System Invoke");

        let render_texture = crate::bindings::cuda::RenderTexture {
            data: unsafe { std::mem::transmute(render_texture_buffer_ptr) },
            size: [globals.render_texture_size[0] as _, globals.render_texture_size[1] as _],
        };

        unsafe {
            self
                .func_compute_render
                .clone()
                .launch(
                    LaunchConfig {
                        block_dim: (crate::bindings::cuda::BLOCK_SIZE as usize as u32, 1, 1),
                        grid_dim: (
                            (render_width * render_height) as u32
                                / (crate::bindings::cuda::BLOCK_SIZE as usize as u32),
                            1,
                            1,
                        ),
                        shared_mem_bytes: 0,
                    },
                    (
                        render_texture.clone(),
                        globals.clone(),
                        camera.clone(),
                    ),
                )
                .unwrap()
        };

        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(
                target_image,
                render_texture_buffer_ptr,
            ).unwrap()
        };

        nvtx::range_end!(range_id);
    }
}
