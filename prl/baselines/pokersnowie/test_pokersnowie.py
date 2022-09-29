from prl.baselines.supervised_learning.data_acquisition.core.parser import Blind, PlayerStack, PlayerWithCards


def test_convert_seats():
    # Arrange
    player_stacks = [PlayerStack(seat_display_name='Seat 1', player_name='SirMarned', stack='$20'),
                     PlayerStack(seat_display_name='Seat 2', player_name='max21rus1988', stack='$50.25'),
                     PlayerStack(seat_display_name='Seat 3', player_name='Communist654', stack='$50'),
                     PlayerStack(seat_display_name='Seat 5', player_name='Becks Baker', stack='$50')]
    hero_name = 'SirMarned'

    expected_ret = 'Seat 0 hero 20\nSeat 1 snowie2 50.25\nSeat 2 snowie3 50\nSeat 4 snowie4 50\n'
    expected_p_dict = {'SirMarned': 'hero',
                       'max21rus1988': 'snowie2',
                       'Communist654': 'snowie3',
                       'Becks Baker': 'snowie4'}
    # Act
    ret = ""
    ith_snowie_player = 1
    player_names_dict = {}
    for p in player_stacks:
        seat_id = int(p.seat_display_name[-1]) - 1
        name = "hero" if hero_name == p.player_name else f"snowie{ith_snowie_player}"
        stack = p.stack[1:]
        ret += f"Seat {seat_id} {name} {stack}\n"
        player_names_dict[p.player_name] = name
        ith_snowie_player += 1
    # Assert
    assert expected_ret == ret
    assert expected_p_dict == player_names_dict


def test_convert_blinds():
    # Arrange
    blinds = [Blind(player_name='max21rus1988', type='small blind', amount='$0.25'),
              Blind(player_name='Communist654', type='big blind', amount='$0.50')]
    player_names_dict = {'SirMarned': 'SirMarned',
                         'max21rus1988': 'snowie2',
                         'Communist654': 'snowie3',
                         'Becks Baker': 'snowie4'}
    expect = "SmallBlind: snowie2 0.25\nBigBlind: snowie3 0.50\n"

    # Act
    sb = blinds[0]
    bb = blinds[1]
    sb_name = player_names_dict[sb.player_name]
    bb_name = player_names_dict[bb.player_name]
    computed = f"SmallBlind: {sb_name} {sb.amount[1:]}\nBigBlind: {bb_name} {bb.amount[1:]}\n"

    # Assert
    assert expect == computed


def test_convert_dealt_cards():
    # Arrange
    showdown_hands = [PlayerWithCards(name='SirMarned', cards='[Js Ad]'),
                      PlayerWithCards(name='Becks Baker', cards='[8d Kd]')]
    player_names_dict = {'SirMarned': 'hero',
                         'max21rus1988': 'snowie2',
                         'Communist654': 'snowie3',
                         'Becks Baker': 'snowie4'}
    # Act
    # Assert
    pass


def test_convert_community_cards():
    # Arrange
    # Act
    # Assert
    pass


def test_convert_moves():
    # Arrange
    # Act
    # Assert
    pass


def test_get_maybe_uncalled_bet():
    # Arrange
    # Act
    # Assert
    pass


if __name__ == "__main__":
    test_convert_seats()
    test_convert_blinds()
    test_convert_dealt_cards()
    test_convert_community_cards()
    test_convert_moves()
    test_get_maybe_uncalled_bet()
