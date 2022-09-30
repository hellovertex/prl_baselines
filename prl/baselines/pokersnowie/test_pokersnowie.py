from prl.baselines.supervised_learning.data_acquisition.core.parser import Blind, PlayerStack, PlayerWithCards, \
    ActionType, Action


def whoami(f):
    print(f"Running {f.__name__}...\n")
    return f


@whoami
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


@whoami
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


@whoami
def test_convert_dealt_cards():
    # Arrange
    showdown_hands = [PlayerWithCards(name='SirMarned', cards='[Js Ad]'),
                      PlayerWithCards(name='Becks Baker', cards='[8d Kd]')]
    player_names_dict = {'SirMarned': 'hero',
                         'max21rus1988': 'snowie2',
                         'Communist654': 'snowie3',
                         'Becks Baker': 'snowie4'}
    expected = "Dealt Cards: [JsAd]\n"

    # Act
    cards = "Dealt Cards: "
    for k, v in player_names_dict.items():
        if v == 'hero':
            for player in showdown_hands:
                if player.name == k:
                    cards += player.cards.replace(" ", "") + "\n"
                    break
    # Assert
    assert expected == cards


@whoami
def test_convert_community_cards():
    # Arrange
    board_cards = '[2s 2d 9h 9c Ac]'
    expect = {'flop': 'FLOP Community Cards:[2s 2d 9h]\n',
              'turn': 'TURN Community Cards:[2s 2d 9h 9c]\n',
              'river': 'RIVER Community Cards:[2s 2d 9h 9c Ac]\n'}
    # Act
    community_cards = {'flop': 'FLOP Community Cards:[' + board_cards[1:9] + ']\n',
                       'turn': 'TURN Community Cards:[' + board_cards[1:12] + ']\n',
                       'river': 'RIVER Community Cards:[' + board_cards[1:15] + ']\n'}
    # Assert
    assert expect == community_cards


@whoami
def test_convert_moves():
    # Arrange
    actions_total = {'preflop': [
        Action(stage='preflop', player_name='Becks Baker', action_type=ActionType.RAISE, raise_amount='1.25'),
        Action(stage='preflop', player_name='SirMarned', action_type=ActionType.CHECK_CALL, raise_amount='1.25'),
        Action(stage='preflop', player_name='max21rus1988', action_type=ActionType.FOLD, raise_amount=-1),
        Action(stage='preflop', player_name='Communist654', action_type=ActionType.FOLD, raise_amount=-1)],
        'flop': [Action(stage='flop', player_name='Becks Baker', action_type=ActionType.RAISE,
                        raise_amount='1.02'),
                 Action(stage='flop', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                        raise_amount='1.02')],
        'turn': [Action(stage='turn', player_name='Becks Baker', action_type=ActionType.CHECK_CALL,
                        raise_amount=-1),
                 Action(stage='turn', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                        raise_amount=-1)],
        'river': [Action(stage='river', player_name='Becks Baker', action_type=ActionType.CHECK_CALL,
                         raise_amount=-1),
                  Action(stage='river', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                         raise_amount=-1)],
        'as_sequence': [Action(stage='preflop', player_name='Becks Baker', action_type=ActionType.RAISE,
                               raise_amount='1.25'),
                        Action(stage='preflop', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                               raise_amount='1.25'),
                        Action(stage='preflop', player_name='max21rus1988', action_type=ActionType.FOLD,
                               raise_amount=-1),
                        Action(stage='preflop', player_name='Communist654', action_type=ActionType.FOLD,
                               raise_amount=-1),
                        Action(stage='flop', player_name='Becks Baker', action_type=ActionType.RAISE,
                               raise_amount='1.02'),
                        Action(stage='flop', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                               raise_amount='1.02'),
                        Action(stage='turn', player_name='Becks Baker', action_type=ActionType.CHECK_CALL,
                               raise_amount=-1),
                        Action(stage='turn', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                               raise_amount=-1),
                        Action(stage='river', player_name='Becks Baker', action_type=ActionType.CHECK_CALL,
                               raise_amount=-1),
                        Action(stage='river', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                               raise_amount=-1)]}
    action_sequence = actions_total['as_sequence']
    player_names_dict = {'SirMarned': 'hero',
                         'max21rus1988': 'snowie2',
                         'Communist654': 'snowie3',
                         'Becks Baker': 'snowie4'}
    expect = {
        'preflop': 'Move: snowie4 raise_bet 1.25\nMove: hero call_check 1.25\nMove: snowie2 folds 0\nMove: snowie3 folds 0\n',
        'flop': 'Move: snowie4 raise_bet 1.02\nMove: hero call_check 1.02\n',
        'turn': 'Move: snowie4 call_check 0\nMove: hero call_check 0\n',
        'river': 'Move: snowie4 call_check 0\nMove: hero call_check 0\n'}

    # Act
    moves = {'preflop': '',
             'flop': '',
             'turn': '',
             'river': ''}
    for a in action_sequence:
        p_name = player_names_dict[a.player_name]
        move = ['folds', 'call_check', 'raise_bet'][a.action_type]
        amt = str(a.raise_amount) if float(a.raise_amount) > 0 else '0'
        moves[a.stage] += f'Move: {p_name} {move} {amt}\n'
    # Assert
    assert expect == moves


@whoami
def test_convert_winners():
    # Arrange
    actions_total = {'preflop': [
        Action(stage='preflop', player_name='Becks Baker', action_type=ActionType.RAISE, raise_amount='1.25'),
        Action(stage='preflop', player_name='SirMarned', action_type=ActionType.CHECK_CALL, raise_amount='1.25'),
        Action(stage='preflop', player_name='max21rus1988', action_type=ActionType.FOLD, raise_amount=-1),
        Action(stage='preflop', player_name='Communist654', action_type=ActionType.FOLD, raise_amount=-1)],
        'flop': [Action(stage='flop', player_name='Becks Baker', action_type=ActionType.RAISE,
                        raise_amount='1.02'),
                 Action(stage='flop', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                        raise_amount='1.02')],
        'turn': [Action(stage='turn', player_name='Becks Baker', action_type=ActionType.CHECK_CALL,
                        raise_amount=-1),
                 Action(stage='turn', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                        raise_amount=-1)],
        'river': [Action(stage='river', player_name='Becks Baker', action_type=ActionType.CHECK_CALL,
                         raise_amount=-1),
                  Action(stage='river', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                         raise_amount=-1)],
        'as_sequence': [Action(stage='preflop', player_name='Becks Baker', action_type=ActionType.RAISE,
                               raise_amount='1.25'),
                        Action(stage='preflop', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                               raise_amount='1.25'),
                        Action(stage='preflop', player_name='max21rus1988', action_type=ActionType.FOLD,
                               raise_amount=-1),
                        Action(stage='preflop', player_name='Communist654', action_type=ActionType.FOLD,
                               raise_amount=-1),
                        Action(stage='flop', player_name='Becks Baker', action_type=ActionType.RAISE,
                               raise_amount='1.02'),
                        Action(stage='flop', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                               raise_amount='1.02'),
                        Action(stage='turn', player_name='Becks Baker', action_type=ActionType.CHECK_CALL,
                               raise_amount=-1),
                        Action(stage='turn', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                               raise_amount=-1),
                        Action(stage='river', player_name='Becks Baker', action_type=ActionType.CHECK_CALL,
                               raise_amount=-1),
                        Action(stage='river', player_name='SirMarned', action_type=ActionType.CHECK_CALL,
                               raise_amount=-1)]}
    action_sequence = actions_total['as_sequence']
    blinds = [Blind(player_name='max21rus1988', type='small blind', amount='$0.25'),
              Blind(player_name='Communist654', type='big blind', amount='$0.50')]
    player_names_dict = {'SirMarned': 'hero',
                         'max21rus1988': 'snowie2',
                         'Communist654': 'snowie3',
                         'Becks Baker': 'snowie4'}
    showdown_players = [PlayerWithCards(name='SirMarned', cards='[Js Ad]'),
                        PlayerWithCards(name='Becks Baker', cards='[8d Kd]')]
    winners = [PlayerWithCards(name='SirMarned', cards='[Js Ad]')]
    expect = "Showdown: hero [Js Ad]\nShowdown: snowie4 [8d Kd]\nWinner: hero 5.29\n"
    # Act
    player_money_in_pot = {}
    for name in player_names_dict.values():
        player_money_in_pot[name] = 0

    total_pot = 0

    # add blinds
    for blind in blinds:
        p_name = player_names_dict[blind.player_name]
        amount = round(float(blind.amount[1:]), 2)
        player_money_in_pot[p_name] += amount
        total_pot += amount

    for a in action_sequence:
        p_name = player_names_dict[a.player_name]
        if float(a.raise_amount) > 0:
            player_money_in_pot[p_name] += float(a.raise_amount)
            total_pot += float(a.raise_amount)
    biggest_contributor = max(player_money_in_pot, key=player_money_in_pot.get)
    biggest_contribution = player_money_in_pot.pop(biggest_contributor)
    second_biggest_or = max(player_money_in_pot, key=player_money_in_pot.get)
    second_biggest_tion = player_money_in_pot[second_biggest_or]
    result = ""
    if biggest_contribution > second_biggest_tion:
        diff = round(biggest_contribution - second_biggest_tion, 2)
        result += f"Move: {biggest_contributor} uncalled_bet {diff}\nWinner: {biggest_contributor} {round(total_pot, 2) - diff}\n"
    else:  # showdown
        for showdown_hand in showdown_players:
            p_name = player_names_dict[showdown_hand.name]
            cards = showdown_hand.cards
            result += f"Showdown: {p_name} {cards}\n"
    for winner in winners:
        result += f"Winner: {player_names_dict[winner.name]} {round(total_pot, 2)}\n"
    # Assert
    assert expect == result


def parse_num(num: str):
    # parse string represenation of float, such that
    # it is rounded at most two digits
    # but only to non-zero decimal places
    # parse float
    num = round(float(num), 2)
    num = str(num).rstrip("0")
    if num.endswith("."):
        num = num[:].rstrip(".")
    return num



def test_parse_numbers():
    num1 = '0.2500'
    assert parse_num(num1) == '0.25'
    num2 = '0.25'
    assert parse_num(num2) == '0.25'
    num3 = '0.200'
    assert parse_num(num3) == '0.2'
    num4 = '0.2'
    assert parse_num(num4) == '0.2'
    num5 = '2.2500'
    assert parse_num(num5) == '2.25'
    num6 = '2.25'
    assert parse_num(num6) == '2.25'
    num7 = '2.200'
    assert parse_num(num7) == '2.2'
    num8 = '2.2'
    assert parse_num(num8) == '2.2'
    num9 = '2.00'
    assert parse_num(num9) == '2'


if __name__ == "__main__":
    test_convert_seats()
    test_convert_blinds()
    test_convert_dealt_cards()
    test_convert_community_cards()
    test_convert_moves()
    test_convert_winners()
    test_parse_numbers()
