use crate::example_scene::ExampleSceneSettings;
use crate::renderer::RenderSettings;
use bevy::{app::AppExit, prelude::*};
use bevy_flycam::MovementSettings;

pub fn receive_input(
    mut movement_settings: ResMut<MovementSettings>,
    mut e_scene_settings: ResMut<ExampleSceneSettings>,
    mut render_settings: ResMut<RenderSettings>,
    keyboard_input: Res<Input<KeyCode>>,
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
}
