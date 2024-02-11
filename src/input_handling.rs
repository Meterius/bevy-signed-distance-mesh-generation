use crate::example_scene::ExampleSceneSettings;
use crate::renderer::{AdvanceMeshGenerationEvent, FinalizeMeshGenerationEvent, RenderSettings};
use bevy::{app::AppExit, prelude::*};
use bevy_flycam::MovementSettings;

pub fn receive_input(
    mut movement_settings: ResMut<MovementSettings>,
    mut e_scene_settings: ResMut<ExampleSceneSettings>,
    mut render_settings: ResMut<RenderSettings>,
    keyboard_input: Res<Input<KeyCode>>,
    mut adv_ew: EventWriter<AdvanceMeshGenerationEvent>,
    mut fin_ew: EventWriter<FinalizeMeshGenerationEvent>,
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
        render_settings.show_partition = !render_settings.show_partition;
    }

    if keyboard_input.just_pressed(KeyCode::B) {
        adv_ew.send(AdvanceMeshGenerationEvent::default());
    }

    if keyboard_input.just_pressed(KeyCode::V) {
        fin_ew.send(FinalizeMeshGenerationEvent::default());
    }
}
