def handle_player_collision(p1, p2):
    p1.velocity += p2.velocity
    p1.velocity *= 0.5
    p2.velocity = p1.velocity.copy()


def handle_kick(p, b):
    b_movement = p.velocity * 1.25
    p.velocity += b.velocity * 0.25
    return b_movement
