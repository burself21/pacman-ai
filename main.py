import pygame
from sys import exit
from board import Board
from game_state import GameState
from aiplayer import AIPlayer
from defaultplayer import DefaultPlayer
import asyncio

pygame.init()

WIDTH = 900
HEIGHT = 950
fps = 60
font_small = pygame.font.Font('freesansbold.ttf', 20)
font_large = pygame.font.Font('freesansbold.ttf', 28)
primary_color = 'blue'
secondary_color = 'white'
background_color = 'black'

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

NUM_PLAYER_IMAGES = 4
NUM_GHOSTS = 4
PLAYER_IMAGE_SIZE = (35, 35)
GHOST_IMAGE_SIZE = (35, 35)

player_images = [pygame.transform.scale(pygame.image.load(f"./graphics/player/{i}.png").convert_alpha(), PLAYER_IMAGE_SIZE) for i in range(NUM_PLAYER_IMAGES)]
ghost_images = [pygame.transform.scale(pygame.image.load(f"./graphics/ghosts/{i}.png").convert_alpha(), GHOST_IMAGE_SIZE) for i in range(NUM_GHOSTS)]
ghost_images.append(pygame.transform.scale(pygame.image.load("./graphics/ghosts/scared.png").convert_alpha(), GHOST_IMAGE_SIZE))  # scared ghost
ghost_images.append(pygame.transform.scale(pygame.image.load("./graphics/ghosts/dead.png").convert_alpha(), GHOST_IMAGE_SIZE))  # dead ghost

PLAYER_SPEED = 2
#GHOST_SPEEDS = (1.95, 1, 1.3, 1.7)
GHOST_SPEEDS = (1.98, 1.98, 1.98, 1.98)
board = Board(WIDTH, HEIGHT, GHOST_SPEEDS)
tile_height = board.tile_height
tile_width = board.tile_width

restart_rect = pygame.Rect(WIDTH // 2 - 90, HEIGHT // 2 + 10, 180, 60)

game_state = GameState(NUM_PLAYER_IMAGES)
#player = DefaultPlayer()
player = AIPlayer()


def move_player(board):
    if board.player_blocked(player.direction):
        player.query_direction_command(actions=board.get_valid_actions(), state=board.get_state())
        if board.player_blocked(player.direction_command):
            return (0, False)
        else:
            player.direction = player.direction_command
    if player.direction == 0:
        player.update_offset((PLAYER_SPEED, 0))
        if player.offset[0] >= tile_width:
            board.update_player_position(1, 0)
            player.offset[0] = 0
            if not board.player_blocked(player.direction_command):
                player.query_direction_command(actions=board.get_valid_actions(), state=board.get_state())
                player.direction = player.direction_command
            return board.eat()
    elif player.direction == 1:
        player.update_offset((-PLAYER_SPEED, 0))
        if player.offset[0] <= -tile_width:
            board.update_player_position(-1, 0)
            player.offset[0] = 0
            if not board.player_blocked(player.direction_command):
                player.query_direction_command(actions=board.get_valid_actions(), state=board.get_state())
                player.direction = player.direction_command
            return board.eat()
    elif player.direction == 2:
        player.update_offset((0, -PLAYER_SPEED))
        if player.offset[1] <= -tile_height:
            board.update_player_position(0, -1)
            player.offset[1] = 0
            if not board.player_blocked(player.direction_command):
                player.query_direction_command(actions=board.get_valid_actions(), state=board.get_state())
                player.direction = player.direction_command
            return board.eat()
    elif player.direction == 3:
        player.update_offset((0, PLAYER_SPEED))
        if player.offset[1] >= tile_height:
            board.update_player_position(0, 1)
            player.offset[1] = 0
            if not board.player_blocked(player.direction_command):
                player.query_direction_command(actions=board.get_valid_actions(), state=board.get_state())
                player.direction = player.direction_command
            return board.eat()
    
    return (0, False)

def check_collisions(board):
    player_pos = board.player_position(offset=player.offset)
    for ghost in board.ghosts:
        ghost_pos = ghost.get_position()
        if abs(ghost_pos[0] - player_pos[0]) < 20 and abs(ghost_pos[1] - player_pos[1]) < 20:
            if ghost.scared and not ghost.eaten:
                ghost.get_eaten()
                game_state.increment_score(50)
            elif not ghost.eaten:
                # player death
                game_state.lose()

def draw_board(board):
    for i in range(board.BOARD_HEIGHT):
        for j in range(board.BOARD_WIDTH):
            tile_value = board.board_values[i][j]
            if tile_value == 1:
                pygame.draw.circle(screen, secondary_color, ((j + 0.5) * tile_width, (i + 0.5) * tile_height), 4)
            elif tile_value == 2 and not game_state.flicker:
                pygame.draw.circle(screen, secondary_color, ((j + 0.5) * tile_width, (i + 0.5) * tile_height), 10)
            elif tile_value == 3:
                pygame.draw.line(screen, primary_color, ((j + 0.5) * tile_width, i * tile_height), 
                                                            ((j + 0.5) * tile_width, (i + 1) * tile_height), 3)
            elif tile_value == 4:
                pygame.draw.line(screen, primary_color, (j * tile_width, (i + 0.5) * tile_height), 
                                                            ((j + 1) * tile_width, (i + 0.5) * tile_height), 3)
            elif tile_value == 5:
                #TR Corner
                #pygame.draw.arc(screen, primary_color, pygame.Rect((j - 0.5) * tile_width, (i + 0.5) * tile_height, tile_width, tile_height), 0, PI / 2, 3)
                pygame.draw.line(screen, primary_color, (j * tile_width, (i + 0.5) * tile_height), ((j + 0.5) * tile_width, (i + 0.5) * tile_height), 3)
                pygame.draw.line(screen, primary_color, ((j + 0.5) * tile_width, (i + 0.5) * tile_height), ((j + 0.5) * tile_width, (i + 1) * tile_height), 3)
            elif tile_value == 6:
                #TL Corner
                pygame.draw.line(screen, primary_color, ((j + 0.5) * tile_width, (i + 1) * tile_height), ((j + 0.5) * tile_width, (i + 0.5) * tile_height), 3)
                pygame.draw.line(screen, primary_color, ((j + 0.5) * tile_width, (i + 0.5) * tile_height), ((j + 1) * tile_width, (i + 0.5) * tile_height), 3)
            elif tile_value == 7:
                pygame.draw.line(screen, primary_color, ((j + 0.5) * tile_width, i * tile_height), ((j + 0.5) * tile_width, (i + 0.5) * tile_height), 3)
                pygame.draw.line(screen, primary_color, ((j + 0.5) * tile_width, (i + 0.5) * tile_height), ((j + 1) * tile_width, (i + 0.5) * tile_height), 3)
            elif tile_value == 8:
                pygame.draw.line(screen, primary_color, (j * tile_width, (i + 0.5) * tile_height), ((j + 0.5) * tile_width, (i + 0.5) * tile_height), 3)
                pygame.draw.line(screen, primary_color, ((j + 0.5) * tile_width, (i + 0.5) * tile_height), ((j + 0.5) * tile_width, i * tile_height), 3)
            elif tile_value == 9:
                pygame.draw.line(screen, secondary_color, (j * tile_width, (i + 0.5) * tile_height), 
                                                            ((j + 1) * tile_width, (i + 0.5) * tile_height), 3)
            else:
                #raise error?
                pass

            
            """match board.board_values[i][j]:
                case 0:
                    # draw nothing, it's blank
                    pass
                case 1:
                    # small dot
                    pygame.draw.circle(screen, secondary_color, ((j + 0.5) * tile_width, (i + 0.5) * tile_height), 4)
                case 2:
                    # big dot
                    pygame.draw.circle(screen, secondary_color, ((j + 0.5) * tile_width, (i + 0.5) * tile_height), 10)
                case 3:
                    pygame.draw.line(screen, primary_color, ((j + 0.5) * tile_width, i * tile_height), 
                                                            ((j + 0.5) * tile_width, (i + 1) * tile_height), 3)
                case 4:
                    pygame.draw.line(screen, primary_color, (j * tile_width, (i + 0.5) * tile_height), 
                                                            ((j + 1) * tile_width, (i + 0.5) * tile_height), 3)
                case 5:
                    pygame.draw.arc(screen, primary_color, pygame.Rect((j - 0.5) * tile_width, (i + 0.5) * tile_width, tile_width, tile_height))
                case 6:
                    pass
                case 7:
                    pass
                case 8:
                    pass
                case 9:
                    pygame.draw.line(screen, secondary_color, (j * tile_width, (i + 0.5) * tile_height), 
                                                            ((j + 1) * tile_width, (i + 0.5) * tile_height), 3)
                case _:
                    print("Confusion... maybe throw an error")"""
                
def draw_player(board):
    # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
    if player.direction == 0:
        screen.blit(player_images[game_state.player_image_counter // 5], board.player_position(offset=player.offset))
    if player.direction == 1:
        screen.blit(pygame.transform.flip(player_images[game_state.player_image_counter // 5], True, False), board.player_position(offset=player.offset))
    if player.direction == 2:
        screen.blit(pygame.transform.rotate(player_images[game_state.player_image_counter // 5], 90), board.player_position(offset=player.offset))
    if player.direction == 3:
        screen.blit(pygame.transform.rotate(player_images[game_state.player_image_counter // 5], 270), board.player_position(offset=player.offset))
    
def draw_ghosts(board):
    for ghost in board.ghosts:
        screen.blit(ghost_images[ghost.get_image_index()], ghost.get_position())

def draw_score():
    score_text = font_small.render(f"Score: {game_state.score}", True, secondary_color)
    screen.blit(score_text, (10, HEIGHT - 30))

def draw_powerup_timer(board):
    if board.powerup_counter > 0:
        timer_text = font_large.render(str(10 - board.powerup_counter // 60), True, primary_color)
        screen.blit(timer_text, (150, HEIGHT - 35))

def draw_footer(board):
    draw_score()
    draw_powerup_timer(board)

def draw_game_over_modal():
    width = 400
    height = 300
    left = (WIDTH - width) // 2
    top = (HEIGHT - height) // 2
    pygame.draw.rect(screen, secondary_color, [left, top, width, height], 0, 10)
    pygame.draw.rect(screen, 'dark gray', [left + 10, top + 10, width - 20, height - 20], 0, 10)
    gameover_string = "You Win!" if game_state.game_won else "Game Over!"
    gameover_text = font_large.render(gameover_string, True, secondary_color)
    screen.blit(gameover_text, (WIDTH // 2 - 85 + 20 * game_state.game_won, top + 60))
    
    pygame.draw.rect(screen, primary_color, [WIDTH // 2 - 90, HEIGHT // 2 + 10, 180, 60], 0, 10)
    play_again_text = font_large.render("Play Again", True, secondary_color)
    screen.blit(play_again_text, (WIDTH // 2 - 75, top + 160 + 15))

def main():
    # globals
    # global run
    # global WIDTH
    # global HEIGHT
    # global fps
    # global font_small
    # global font_large
    # global primary_color
    # global secondary_color
    # global background_color

    # global counter
    # global player_image_counter
    # global flicker
    # global game_over
    # global game_won

    # global screen
    # global clock

    # global NUM_PLAYER_IMAGES
    # global NUM_GHOST
    # global PLAYER_IMAGE_SIZE
    # global GHOST_IMAGE_SIZE

    # global player_images
    # global ghost_images


    # global PLAYER_SPEED
    # global GHOST_SPEEDS
    # global board
    # global tile_height
    # global tile_width
    # global player_offset_x
    # global player_offset_y
    # global direction
    # global direction_command
    # global score

    # global restart_rect

    # counter = 0
    # player_image_counter = 0
    # flicker = False
    # game_over = True
    # game_won = True

    # player_offset_x = 0
    # player_offset_y = 0
    # direction = 0          # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
    # update_direction_command(0
    # score = 0

    run = True
    i = 0
    [print(arr.sum()) for arr in board.get_state()[0]]
    while run:
        clock.tick(60)
        board.tick()
        game_state.tick()
        
        screen.fill(background_color)
        draw_board(board)
        if not game_state.game_over:
            

            points, win = move_player(board)
            i += 1
            print("Step i:")
            print("Player:", (board.pac_x, board.pac_y))
            print("Ghosts:")
            for ghost in board.ghosts:
                print(ghost.index, ghost.coords)
            board.move_ghosts()
            game_state.increment_score(points)
            if win:
                game_state.win()
        draw_player(board)
        draw_ghosts(board)
        draw_footer(board)
        if game_state.game_over:
            draw_game_over_modal()
            # check_restart_click
            pos = pygame.mouse.get_pos()
            if pygame.mouse.get_pressed()[0] and restart_rect.collidepoint(pos):
                # reset everything
                board.reset()
            
                # reset everything else
                game_state.reset()
                player.reset()
                
                



        check_collisions(board)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # turn commands are processed once we get to the next tile
                if event.key == pygame.K_RIGHT:
                    player.update_direction_command(0)
                if event.key == pygame.K_LEFT:
                    player.update_direction_command(1)
                if event.key == pygame.K_UP:
                    player.update_direction_command(2)
                if event.key == pygame.K_DOWN:
                    player.update_direction_command(3)
            if event.type == pygame.QUIT:
                run = False
                

        # update everything
        pygame.display.update()
        #await asyncio.sleep(0)

    pygame.quit()
    exit()

#asyncio.run(main())
if __name__ == "__main__":
    main()