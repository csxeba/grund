import unittest

import numpy as np

from grund.dodge.environment import (Dodge, DodgeConfig, _get_colliding_balls,
                                     _get_colliding_pairs)


class TestDodgeInitialization(unittest.TestCase):
    def setUp(self):
        self.config = DodgeConfig(
            simulation_width=64,
            simulation_height=64,
            num_enemy_balls=4,
            ball_radius=5,
            ball_velocity=5,
        )
        self.simulation = Dodge(config=self.config)

    def test_ball_positions_within_bounds(self):
        self.simulation.reset()
        self.assertTrue(
            np.all(self.simulation.ball_positions >= self.config.ball_radius)
        )
        self.assertTrue(
            np.all(
                self.simulation.ball_positions[:, 0]
                <= self.config.simulation_width - self.config.ball_radius
            )
        )
        self.assertTrue(
            np.all(
                self.simulation.ball_positions[:, 1]
                <= self.config.simulation_height - self.config.ball_radius
            )
        )

    def test_no_overlapping_balls(self):
        self.simulation.reset()
        colliding_balls = _get_colliding_balls(
            ball_positions=self.simulation.ball_positions,
            ball_radius=self.config.ball_radius,
        )
        self.assertEqual(len(colliding_balls), 0)

    def test_ball_directions_initialized(self):
        self.simulation.reset()
        self.assertEqual(
            self.simulation.ball_directions.shape, (self.config.num_enemy_balls, 2)
        )
        self.assertTrue(
            np.all(
                np.logical_and(
                    self.simulation.ball_directions >= -1,
                    self.simulation.ball_directions <= 1,
                )
            )
        )

    def test_fail_on_small_simulation_size(self):
        small_config = DodgeConfig(
            simulation_width=16,
            simulation_height=16,
            num_enemy_balls=2,
            ball_radius=6,
            ball_velocity=1,
        )
        small_simulation = Dodge(config=small_config)
        with self.assertRaises(RuntimeError):
            small_simulation.reset()


class TestDodgeStep(unittest.TestCase):
    def setUp(self):
        self.config = DodgeConfig(
            simulation_width=64,
            simulation_height=64,
            num_enemy_balls=4,
            ball_radius=5,
            ball_velocity=5,
        )
        self.simulation = Dodge(config=self.config)

    def test_step_no_collision(self):
        # Setup a scenario where there are no initial collisions
        self.simulation.ball_positions = np.array(
            [[10, 10], [40, 10], [10, 40], [40, 40]]
        )
        self.simulation.ball_directions = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])

        # Execute one step
        self.simulation.step()

        # Expected positions after one step, assuming ball_velocity = 5
        expected_positions = np.array([[10, 15], [45, 10], [5, 40], [40, 35]])
        np.testing.assert_array_equal(
            self.simulation.ball_positions,
            expected_positions,
            "Ball positions are not updated correctly",
        )

    def test_step_with_collision(self):
        # Setup a scenario where there is an initial collision
        self.simulation.ball_positions = np.array(
            [[10, 10], [15, 10], [50, 50], [60, 60]]
        )
        self.simulation.ball_directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

        # Execute one step
        self.simulation.step()

        # Expected positions after one step, assuming ball_velocity = 5
        # Balls 0 and 1 should reflect their directions after collision
        expected_positions = np.array([[5, 10], [20, 10], [50, 55], [60, 55]])
        expected_directions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])

        np.testing.assert_array_equal(
            self.simulation.ball_positions,
            expected_positions,
            "Ball positions are not updated correctly",
        )
        np.testing.assert_array_almost_equal(
            self.simulation.ball_directions,
            expected_directions,
            decimal=6,
            err_msg="Ball directions are not updated correctly",
        )


class TestDodgeUtilities(unittest.TestCase):
    def test_get_colliding_pairs(self):
        # Setup a scenario where there are known collisions
        ball_positions = np.array([[10, 10], [22, 10], [50, 50], [35, 35]])
        ball_radius = 10

        colliding_pairs = _get_colliding_pairs(
            ball_positions=ball_positions, ball_radius=ball_radius
        )
        expected_colliding_pairs = np.array([[0, 1]])

        np.testing.assert_array_equal(
            colliding_pairs,
            expected_colliding_pairs,
            "Colliding pairs are not as expected",
        )


if __name__ == "__main__":
    unittest.main()
