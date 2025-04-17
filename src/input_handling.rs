use crate::example_scene::ExampleSceneSettings;
use crate::renderer::{MeshGenRefineEvent, MeshGenAdvanceEvent, RenderSettings};
use bevy::{app::AppExit, prelude::*};
use bevy_flycam::MovementSettings;

pub fn receive_input(
    mut movement_settings: ResMut<MovementSettings>,
    mut e_scene_settings: ResMut<ExampleSceneSettings>,
    mut render_settings: ResMut<RenderSettings>,
    keyboard_input: Res<Input<KeyCode>>,
    mut adv_ew: EventWriter<MeshGenRefineEvent>,
    mut fin_ew: EventWriter<MeshGenAdvanceEvent>,
    mut exit: EventWriter<AppExit>,
) {
    if keyboard_input.just_pressed(KeyCode::Escape) {
        exit.send(AppExit);
    }

    if keyboard_input.just_pressed(KeyCode::ControlLeft) {
        movement_settings.speed = 200.0;
    } else if keyboard_input.just_released(KeyCode::ControlLeft) {
        movement_settings.speed = 12.0;
    }

    if keyboard_input.just_pressed(KeyCode::M) {
        e_scene_settings.enable_movement = !e_scene_settings.enable_movement;
    }

    if keyboard_input.just_pressed(KeyCode::N) {
        render_settings.show_preview = !render_settings.show_preview;
    }

    if keyboard_input.just_pressed(KeyCode::K) {
        render_settings.show_wireframe = !render_settings.show_wireframe;
    }

    if keyboard_input.just_pressed(KeyCode::B) {
        adv_ew.send(MeshGenRefineEvent::default());
    }

    if keyboard_input.just_pressed(KeyCode::V) {
        fin_ew.send(MeshGenAdvanceEvent::default());
    }
}
