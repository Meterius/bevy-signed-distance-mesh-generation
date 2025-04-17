use crate::example_scene::ExampleScenePlugin;
use crate::renderer::{MeshGenRefineEvent, MeshGenAdvanceEvent};
use bevy::app::AppExit;
use bevy::diagnostic::{EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::pbr::wireframe::WireframeConfig;
use bevy::prelude::*;
use bevy::render::settings::{Backends, RenderCreation, WgpuFeatures, WgpuSettings};
use bevy::render::RenderPlugin;
use bevy::window::{CursorGrabMode, PresentMode, PrimaryWindow, WindowResolution};
use bevy_editor_pls::EditorPlugin;
use bevy_flycam::NoCameraPlayerPlugin;
use bevy_obj::ObjPlugin;

pub mod bindings;
pub mod example_scene;
pub mod input_handling;
pub mod renderer;
mod cuda;

fn headless_startup(
    mut ew_adv: EventWriter<MeshGenRefineEvent>,
    mut ew_fin: EventWriter<MeshGenAdvanceEvent>,
    mut ew_exit: EventWriter<AppExit>,
) {
    ew_adv.send(MeshGenRefineEvent::default());

    for _ in 0..2 {
        ew_adv.send(MeshGenRefineEvent::default());
    }

    ew_fin.send(MeshGenAdvanceEvent::default());

    ew_exit.send(AppExit::default());
}

fn main() {
    unsafe { cudarc::driver::sys::cuProfilerStart() };

    let mut app = App::new();

    app
        .insert_resource(Msaa::Sample8)
        .insert_resource(WireframeConfig {
            global: false,
            default_color: Color::RED,
        })
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        position: WindowPosition::Centered(MonitorSelection::Index(0)),
                        present_mode: PresentMode::AutoVsync,
                        resolution: WindowResolution::new(1920., 1080.),
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        features: WgpuFeatures::POLYGON_MODE_LINE | WgpuFeatures::POLYGON_MODE_POINT,
                        backends: Some(Backends::VULKAN),
                        ..default()
                    }),
                }),
            ObjPlugin,
            FrameTimeDiagnosticsPlugin::default(),
            EntityCountDiagnosticsPlugin::default(),
            EditorPlugin::default(),
            renderer::RayMarcherRenderPlugin::default(),
            NoCameraPlayerPlugin,
            ExampleScenePlugin::default(),
        ));

    app.add_systems(
        PostStartup,
        |mut primary_window: Query<&mut Window, With<PrimaryWindow>>,
         mut key_binds: ResMut<bevy_flycam::KeyBindings>| {
            let mut window = primary_window.single_mut();
            window.cursor.grab_mode = CursorGrabMode::None;
            window.cursor.visible = true;
            key_binds.toggle_grab_cursor = KeyCode::F;
        },
    );

    app.add_systems(Startup, example_scene::setup_scene);
    app.add_systems(Update, input_handling::receive_input);

    if std::env::var("HEADLESS").unwrap_or(String::from("")) != "" {
        app.add_systems(Startup, headless_startup);
    }

    app.run();
}
