/// \file grammar.cpp
/// \brief GBNF grammar-constrained sampling — implementation
/// \note  Ported from llama.cpp's llama-grammar.cpp (MIT license).
///        See grammar.hpp for the full attribution and scope notes.
#include "grammar/grammar.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#define MAX_REPETITION_THRESHOLD 2000

// ===== UTF-8 helpers =======================================================

static std::pair<uint32_t, const char *> decode_utf8(const char * src) {
    static const int lookup[] = {1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,4};
    uint8_t  first = static_cast<uint8_t>(*src);
    uint8_t  high  = first >> 4;
    int      len   = lookup[high];
    uint8_t  mask  = (1 << (8 - len)) - 1;
    uint32_t value = first & mask;
    const char * end = src + len;
    const char * pos = src + 1;
    for (; pos < end && *pos; pos++) {
        value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
    }
    return {value, pos};
}

static std::pair<std::vector<uint32_t>, flm_partial_utf8> decode_utf8(
        const std::string & src, flm_partial_utf8 partial_start) {
    static const int lookup[] = {1,1,1,1,1,1,1,1,0,0,0,0,2,2,3,4};
    const char * pos = src.c_str();
    std::vector<uint32_t> code_points;
    code_points.reserve(src.size() + 1);

    uint32_t value    = partial_start.value;
    int      n_remain = partial_start.n_remain;

    while (*pos != 0 && n_remain > 0) {
        uint8_t next = static_cast<uint8_t>(*pos);
        if ((next >> 6) != 2) {
            code_points.push_back(0);
            return {std::move(code_points), {0, -1}};
        }
        value = (value << 6) + (next & 0x3F);
        ++pos; --n_remain;
    }
    if (partial_start.n_remain > 0 && n_remain == 0) {
        code_points.push_back(value);
    }

    while (*pos != 0) {
        uint8_t first = static_cast<uint8_t>(*pos);
        uint8_t high  = first >> 4;
        n_remain = lookup[high] - 1;
        if (n_remain < 0) {
            code_points.clear();
            code_points.push_back(0);
            return {std::move(code_points), {0, n_remain}};
        }
        uint8_t mask = (1 << (7 - n_remain)) - 1;
        value = first & mask;
        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos; --n_remain;
        }
        if (n_remain == 0) code_points.push_back(value);
    }
    code_points.push_back(0);
    return {std::move(code_points), {value, n_remain}};
}

// ===== parsing helpers =====================================================

static bool is_digit_char(char c) { return '0' <= c && c <= '9'; }
static bool is_word_char(char c) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || is_digit_char(c);
}

static std::pair<uint32_t, const char *> parse_hex(const char * src, int size) {
    const char * pos = src;
    const char * end = src + size;
    uint32_t value = 0;
    for (; pos < end && *pos; pos++) {
        value <<= 4;
        char c = *pos;
        if      ('a' <= c && c <= 'f') value += c - 'a' + 10;
        else if ('A' <= c && c <= 'F') value += c - 'A' + 10;
        else if ('0' <= c && c <= '9') value += c - '0';
        else break;
    }
    if (pos != end) {
        throw std::runtime_error("expecting " + std::to_string(size) + " hex chars at " + src);
    }
    return {value, pos};
}

static const char * parse_space(const char * src, bool newline_ok) {
    const char * pos = src;
    while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
           (newline_ok && (*pos == '\r' || *pos == '\n'))) {
        if (*pos == '#') { while (*pos && *pos != '\r' && *pos != '\n') pos++; }
        else pos++;
    }
    return pos;
}

static const char * parse_name(const char * src) {
    const char * pos = src;
    while (is_word_char(*pos)) pos++;
    if (pos == src) throw std::runtime_error(std::string("expecting name at ") + src);
    return pos;
}

static const char * parse_int(const char * src) {
    const char * pos = src;
    while (is_digit_char(*pos)) pos++;
    if (pos == src) throw std::runtime_error(std::string("expecting integer at ") + src);
    return pos;
}

static std::pair<uint32_t, const char *> parse_char(const char * src) {
    if (*src == '\\') {
        switch (src[1]) {
            case 'x': return parse_hex(src + 2, 2);
            case 'u': return parse_hex(src + 2, 4);
            case 'U': return parse_hex(src + 2, 8);
            case 't': return {'\t', src + 2};
            case 'r': return {'\r', src + 2};
            case 'n': return {'\n', src + 2};
            case '\\': case '"': case '[': case ']':
                return {static_cast<uint32_t>(src[1]), src + 2};
            default:
                throw std::runtime_error(std::string("unknown escape at ") + src);
        }
    } else if (*src) {
        return decode_utf8(src);
    }
    throw std::runtime_error("unexpected end of input");
}

// ===== token parsing (<[id]>) =============================================

static std::pair<uint32_t, const char *> parse_token(const char * src) {
    const char * pos = src;
    if (*pos != '<') throw std::runtime_error(std::string("expecting '<' at ") + pos);
    pos++;
    if (*pos != '[') throw std::runtime_error(std::string("expecting '[' at ") + pos);
    pos++;
    const char * int_end = pos;
    while (is_digit_char(*int_end)) int_end++;
    if (int_end == pos) throw std::runtime_error(std::string("expecting digit at ") + pos);
    uint32_t token_id = std::stoul(std::string(pos, int_end - pos));
    pos = int_end;
    if (*pos != ']') throw std::runtime_error(std::string("expecting ']' at ") + pos);
    pos++;
    if (*pos != '>') throw std::runtime_error(std::string("expecting '>' at ") + pos);
    pos++;
    return {token_id, pos};
}

static bool grammar_match_token(const flm_grammar_element * pos, int token) {
    if (pos->type == FLM_GRETYPE_TOKEN)
        return pos->value == static_cast<uint32_t>(token);
    if (pos->type == FLM_GRETYPE_TOKEN_NOT)
        return pos->value != static_cast<uint32_t>(token);
    return false;
}

// ===== grammar element helpers =============================================

static bool is_end_of_sequence(const flm_grammar_element * pos) {
    return pos->type == FLM_GRETYPE_END || pos->type == FLM_GRETYPE_ALT;
}

static bool is_char_element(flm_grammar_element elem) {
    switch (elem.type) {
        case FLM_GRETYPE_CHAR:
        case FLM_GRETYPE_CHAR_NOT:
        case FLM_GRETYPE_CHAR_ALT:
        case FLM_GRETYPE_CHAR_RNG_UPPER:
        case FLM_GRETYPE_CHAR_ANY:
            return true;
        default:
            return false;
    }
}

static std::pair<bool, const flm_grammar_element *> grammar_match_char(
        const flm_grammar_element * pos, uint32_t chr) {
    bool found = false;
    bool is_positive = pos->type == FLM_GRETYPE_CHAR || pos->type == FLM_GRETYPE_CHAR_ANY;
    assert(is_positive || pos->type == FLM_GRETYPE_CHAR_NOT);

    do {
        if (pos[1].type == FLM_GRETYPE_CHAR_RNG_UPPER) {
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else if (pos->type == FLM_GRETYPE_CHAR_ANY) {
            found = true;
            pos += 1;
        } else {
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == FLM_GRETYPE_CHAR_ALT);

    return {found == is_positive, pos};
}

static bool grammar_match_partial_char(
        const flm_grammar_element * pos, flm_partial_utf8 partial) {
    bool is_positive = pos->type == FLM_GRETYPE_CHAR || pos->type == FLM_GRETYPE_CHAR_ANY;
    assert(is_positive || pos->type == FLM_GRETYPE_CHAR_NOT);

    uint32_t pval    = partial.value;
    int      n_rem   = partial.n_remain;

    if (n_rem < 0 || (n_rem == 1 && pval < 2)) return false;

    uint32_t low  = pval << (n_rem * 6);
    uint32_t high = low | ((1 << (n_rem * 6)) - 1);
    if (low == 0) {
        if (n_rem == 2) low = 1 << 11;
        else if (n_rem == 3) low = 1 << 16;
    }

    do {
        if (pos[1].type == FLM_GRETYPE_CHAR_RNG_UPPER) {
            if (pos->value <= high && low <= pos[1].value) return is_positive;
            pos += 2;
        } else if (pos->type == FLM_GRETYPE_CHAR_ANY) {
            return true;
        } else {
            if (low <= pos->value && pos->value <= high) return is_positive;
            pos += 1;
        }
    } while (pos->type == FLM_GRETYPE_CHAR_ALT);

    return !is_positive;
}

// ===== left-recursion detector =============================================

static bool detect_left_recursion(
        const flm_grammar_rules & rules, size_t rule_index,
        std::vector<bool> & visited, std::vector<bool> & in_progress,
        std::vector<bool> & may_be_empty) {
    if (in_progress[rule_index]) return true;
    in_progress[rule_index] = true;

    const auto & rule = rules[rule_index];
    bool at_start = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (is_end_of_sequence(&rule[i])) {
            if (at_start) { may_be_empty[rule_index] = true; break; }
            at_start = true;
        } else {
            at_start = false;
        }
    }

    bool recurse = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (rule[i].type == FLM_GRETYPE_RULE_REF && recurse) {
            if (detect_left_recursion(rules, rule[i].value, visited, in_progress, may_be_empty))
                return true;
            if (!may_be_empty[rule[i].value]) recurse = false;
        } else if (is_end_of_sequence(&rule[i])) {
            recurse = true;
        } else {
            recurse = false;
        }
    }

    in_progress[rule_index] = false;
    visited[rule_index] = true;
    return false;
}

// ===== flm_grammar_parser implementation ===================================

uint32_t flm_grammar_parser::get_symbol_id(const char * src, size_t len) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    auto result = symbol_ids.emplace(std::string(src, len), next_id);
    return result.first->second;
}

uint32_t flm_grammar_parser::generate_symbol_id(const std::string & base_name) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    symbol_ids[base_name + '_' + std::to_string(next_id)] = next_id;
    return next_id;
}

void flm_grammar_parser::add_rule(uint32_t rule_id, const flm_grammar_rule & rule) {
    if (rules.size() <= rule_id) rules.resize(rule_id + 1);
    rules[rule_id] = rule;
}

const char * flm_grammar_parser::parse_alternates(
        const char * src, const std::string & rule_name,
        uint32_t rule_id, bool is_nested) {
    flm_grammar_rule rule;
    const char * pos = parse_sequence(src, rule_name, rule, is_nested);
    while (*pos == '|') {
        rule.push_back({FLM_GRETYPE_ALT, 0});
        pos = parse_space(pos + 1, true);
        pos = parse_sequence(pos, rule_name, rule, is_nested);
    }
    rule.push_back({FLM_GRETYPE_END, 0});
    add_rule(rule_id, rule);
    return pos;
}

const char * flm_grammar_parser::parse_sequence(
        const char * src, const std::string & rule_name,
        flm_grammar_rule & rule, bool is_nested) {
    size_t last_sym_start = rule.size();
    const char * pos = src;
    uint64_t n_prev_rules = 1;

    auto handle_repetitions = [&](uint64_t min_times, uint64_t max_times) {
        bool no_max = max_times == UINT64_MAX;
        if (last_sym_start == rule.size()) {
            throw std::runtime_error(std::string("expecting preceding item to */+/?/{ at ") + pos);
        }

        flm_grammar_rule prev_rule(rule.begin() + last_sym_start, rule.end());
        uint64_t total_rules = 1;
        if (!no_max && max_times > 0) total_rules = max_times;
        else if (min_times > 0) total_rules = min_times;

        if (n_prev_rules * total_rules >= MAX_REPETITION_THRESHOLD) {
            throw std::runtime_error("repetition count exceeds safe threshold");
        }

        if (min_times == 0) {
            rule.resize(last_sym_start);
        } else {
            for (uint64_t i = 1; i < min_times; i++)
                rule.insert(rule.end(), prev_rule.begin(), prev_rule.end());
        }

        uint32_t last_rec_rule_id = 0;
        auto n_opt = no_max ? 1 : max_times - min_times;

        flm_grammar_rule rec_rule(prev_rule);
        for (uint64_t i = 0; i < n_opt; i++) {
            rec_rule.resize(prev_rule.size());
            uint32_t rec_rule_id = generate_symbol_id(rule_name);
            if (i > 0 || no_max)
                rec_rule.push_back({FLM_GRETYPE_RULE_REF, no_max ? rec_rule_id : last_rec_rule_id});
            rec_rule.push_back({FLM_GRETYPE_ALT, 0});
            rec_rule.push_back({FLM_GRETYPE_END, 0});
            add_rule(rec_rule_id, rec_rule);
            last_rec_rule_id = rec_rule_id;
        }
        if (n_opt > 0) rule.push_back({FLM_GRETYPE_RULE_REF, last_rec_rule_id});
        n_prev_rules *= total_rules;
    };

    while (*pos) {
        if (*pos == '"') {
            pos++;
            last_sym_start = rule.size();
            n_prev_rules = 1;
            while (*pos != '"') {
                if (!*pos) throw std::runtime_error("unexpected end of input");
                auto cp = parse_char(pos);
                pos = cp.second;
                rule.push_back({FLM_GRETYPE_CHAR, cp.first});
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '[') {
            pos++;
            auto start_type = FLM_GRETYPE_CHAR;
            if (*pos == '^') { pos++; start_type = FLM_GRETYPE_CHAR_NOT; }
            last_sym_start = rule.size();
            n_prev_rules = 1;
            while (*pos != ']') {
                if (!*pos) throw std::runtime_error("unexpected end of input");
                auto cp = parse_char(pos);
                pos = cp.second;
                auto type = last_sym_start < rule.size() ? FLM_GRETYPE_CHAR_ALT : start_type;
                rule.push_back({type, cp.first});
                if (pos[0] == '-' && pos[1] != ']') {
                    if (!pos[1]) throw std::runtime_error("unexpected end of input");
                    auto ecp = parse_char(pos + 1);
                    pos = ecp.second;
                    rule.push_back({FLM_GRETYPE_CHAR_RNG_UPPER, ecp.first});
                }
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '<' || *pos == '!') {
            // Token matching: <[id]> or !<[id]>
            auto type = FLM_GRETYPE_TOKEN;
            if (*pos == '!') { type = FLM_GRETYPE_TOKEN_NOT; pos++; }
            auto token_pair = parse_token(pos);
            pos = parse_space(token_pair.second, is_nested);
            last_sym_start = rule.size();
            n_prev_rules = 1;
            rule.push_back({type, token_pair.first});
        } else if (is_word_char(*pos)) {
            const char * name_end = parse_name(pos);
            uint32_t ref_id = get_symbol_id(pos, name_end - pos);
            pos = parse_space(name_end, is_nested);
            last_sym_start = rule.size();
            n_prev_rules = 1;
            rule.push_back({FLM_GRETYPE_RULE_REF, ref_id});
        } else if (*pos == '(') {
            pos = parse_space(pos + 1, true);
            uint32_t n_before = static_cast<uint32_t>(symbol_ids.size());
            uint32_t sub_id = generate_symbol_id(rule_name);
            pos = parse_alternates(pos, rule_name, sub_id, true);
            n_prev_rules = std::max(1u, static_cast<uint32_t>(symbol_ids.size()) - n_before);
            last_sym_start = rule.size();
            rule.push_back({FLM_GRETYPE_RULE_REF, sub_id});
            if (*pos != ')') throw std::runtime_error(std::string("expecting ')' at ") + pos);
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '.') {
            last_sym_start = rule.size();
            n_prev_rules = 1;
            rule.push_back({FLM_GRETYPE_CHAR_ANY, 0});
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '*') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, UINT64_MAX);
        } else if (*pos == '+') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(1, UINT64_MAX);
        } else if (*pos == '?') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, 1);
        } else if (*pos == '{') {
            pos = parse_space(pos + 1, is_nested);
            if (!is_digit_char(*pos))
                throw std::runtime_error(std::string("expecting int at ") + pos);
            const char * ie = parse_int(pos);
            uint64_t min_t = std::stoull(std::string(pos, ie - pos));
            pos = parse_space(ie, is_nested);

            uint64_t max_t = UINT64_MAX;
            if (*pos == '}') {
                max_t = min_t;
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == ',') {
                pos = parse_space(pos + 1, is_nested);
                if (is_digit_char(*pos)) {
                    const char * ie2 = parse_int(pos);
                    max_t = std::stoull(std::string(pos, ie2 - pos));
                    pos = parse_space(ie2, is_nested);
                }
                if (*pos != '}')
                    throw std::runtime_error(std::string("expecting '}' at ") + pos);
                pos = parse_space(pos + 1, is_nested);
            } else {
                throw std::runtime_error(std::string("expecting ',' at ") + pos);
            }
            bool has_max = max_t != UINT64_MAX;
            if (min_t > MAX_REPETITION_THRESHOLD || (has_max && max_t > MAX_REPETITION_THRESHOLD))
                throw std::runtime_error("repetition count exceeds safe threshold");
            handle_repetitions(min_t, max_t);
        } else {
            break;
        }
    }
    return pos;
}

const char * flm_grammar_parser::parse_rule(const char * src) {
    const char * name_end = parse_name(src);
    const char * pos      = parse_space(name_end, false);
    size_t       name_len = name_end - src;
    uint32_t     rule_id  = get_symbol_id(src, name_len);
    const std::string name(src, name_len);

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '='))
        throw std::runtime_error(std::string("expecting ::= at ") + pos);
    pos = parse_space(pos + 3, true);
    pos = parse_alternates(pos, name, rule_id, false);

    if (*pos == '\r') pos += pos[1] == '\n' ? 2 : 1;
    else if (*pos == '\n') pos++;
    else if (*pos) throw std::runtime_error(std::string("expecting newline or end at ") + pos);
    return parse_space(pos, true);
}

bool flm_grammar_parser::parse(const char * src) {
    try {
        const char * pos = parse_space(src, true);
        while (*pos) pos = parse_rule(pos);

        for (const auto & rule : rules) {
            if (rule.empty()) throw std::runtime_error("undefined rule");
            for (const auto & elem : rule) {
                if (elem.type == FLM_GRETYPE_RULE_REF) {
                    if (elem.value >= rules.size() || rules[elem.value].empty()) {
                        for (const auto & kv : symbol_ids) {
                            if (kv.second == elem.value)
                                throw std::runtime_error("undefined rule '" + kv.first + "'");
                        }
                    }
                }
            }
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "flm_grammar: parse error: %s\n", err.what());
        rules.clear();
        return false;
    }
    return true;
}

// ===== FlmGrammar — pushdown automaton =====================================

void FlmGrammar::advance_stack(
        const flm_grammar_rules & rules,
        const flm_grammar_stack & stack,
        flm_grammar_stacks & new_stacks) {
    std::vector<flm_grammar_stack> todo;
    todo.push_back(stack);

    auto stack_cmp = [](const flm_grammar_stack & a, const flm_grammar_stack & b) {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
            [](const flm_grammar_element * pa, const flm_grammar_element * pb) {
                return pa < pb;
            });
    };
    std::set<flm_grammar_stack, decltype(stack_cmp)> seen(stack_cmp);

    while (!todo.empty()) {
        flm_grammar_stack curr = std::move(todo.back());
        todo.pop_back();

        if (seen.count(curr)) continue;
        seen.insert(curr);

        if (curr.empty()) {
            if (std::find(new_stacks.begin(), new_stacks.end(), curr) == new_stacks.end())
                new_stacks.emplace_back(std::move(curr));
            continue;
        }

        const flm_grammar_element * pos = curr.back();
        switch (pos->type) {
        case FLM_GRETYPE_RULE_REF: {
            size_t rule_id = pos->value;
            const flm_grammar_element * subpos = rules[rule_id].data();
            do {
                flm_grammar_stack next(curr.begin(), curr.end() - 1);
                if (!is_end_of_sequence(pos + 1)) next.push_back(pos + 1);
                if (!is_end_of_sequence(subpos))   next.push_back(subpos);
                todo.push_back(std::move(next));
                while (!is_end_of_sequence(subpos)) subpos++;
                if (subpos->type == FLM_GRETYPE_ALT) subpos++;
                else break;
            } while (true);
            break;
        }
        case FLM_GRETYPE_CHAR:
        case FLM_GRETYPE_CHAR_NOT:
        case FLM_GRETYPE_CHAR_ANY:
        case FLM_GRETYPE_TOKEN:
        case FLM_GRETYPE_TOKEN_NOT:
            if (std::find(new_stacks.begin(), new_stacks.end(), curr) == new_stacks.end())
                new_stacks.emplace_back(std::move(curr));
            break;
        default:
            assert(false && "grammar advance_stack: unexpected element type");
            break;
        }
    }
}

flm_grammar_candidates FlmGrammar::reject_candidates_for_stack(
        const flm_grammar_rules & rules,
        const flm_grammar_stack & stack,
        const flm_grammar_candidates & candidates) {
    flm_grammar_candidates rejects;
    rejects.reserve(candidates.size());

    if (stack.empty()) {
        for (const auto & tok : candidates) {
            if (*tok.code_points != 0 || tok.partial_utf8.n_remain != 0)
                rejects.push_back(tok);
        }
        return rejects;
    }

    const flm_grammar_element * stack_pos = stack.back();

    // Token-level matching: if the top of stack is a token rule, match by ID
    if (stack_pos->type == FLM_GRETYPE_TOKEN || stack_pos->type == FLM_GRETYPE_TOKEN_NOT) {
        for (const auto & tok : candidates) {
            if (*tok.code_points == 0) {
                if (tok.partial_utf8.n_remain != 0) rejects.push_back(tok);
            } else if (!grammar_match_token(stack_pos, tok.token_id)) {
                rejects.push_back(tok);
            }
        }
        return rejects;
    }

    flm_grammar_candidates next_candidates;
    next_candidates.reserve(candidates.size());

    for (const auto & tok : candidates) {
        if (*tok.code_points == 0) {
            if (tok.partial_utf8.n_remain != 0 &&
                    !grammar_match_partial_char(stack_pos, tok.partial_utf8))
                rejects.push_back(tok);
        } else if (grammar_match_char(stack_pos, *tok.code_points).first) {
            next_candidates.push_back({tok.index, tok.code_points + 1, tok.token_id, tok.partial_utf8});
        } else {
            rejects.push_back(tok);
        }
    }

    const auto * after = grammar_match_char(stack_pos, 0).second;
    flm_grammar_stack stack_after(stack.begin(), stack.end() - 1);
    if (!is_end_of_sequence(after)) stack_after.push_back(after);

    flm_grammar_stacks next_stacks;
    advance_stack(rules, stack_after, next_stacks);

    auto next_rejects = reject_candidates(rules, next_stacks, next_candidates);
    for (const auto & tok : next_rejects)
        rejects.push_back({tok.index, tok.code_points - 1, tok.token_id, tok.partial_utf8});

    return rejects;
}

flm_grammar_candidates FlmGrammar::reject_candidates(
        const flm_grammar_rules & rules,
        const flm_grammar_stacks & stacks,
        const flm_grammar_candidates & candidates) {
    assert(!stacks.empty());
    if (candidates.empty()) return {};

    auto rejects = reject_candidates_for_stack(rules, stacks.front(), candidates);
    for (size_t i = 1; i < stacks.size(); ++i)
        rejects = reject_candidates_for_stack(rules, stacks[i], rejects);
    return rejects;
}

// ===== FlmGrammar public API ===============================================

std::unique_ptr<FlmGrammar> FlmGrammar::create(
        const std::string & grammar_str,
        const std::string & grammar_root,
        const std::vector<std::string> & token_pieces,
        const std::vector<int> & eog_ids) {

    flm_grammar_parser parser;
    if (!parser.parse(grammar_str.c_str()) || parser.rules.empty()) {
        fprintf(stderr, "flm_grammar: failed to parse grammar\n");
        return nullptr;
    }

    if (parser.symbol_ids.find(grammar_root) == parser.symbol_ids.end()) {
        fprintf(stderr, "flm_grammar: grammar has no '%s' rule\n", grammar_root.c_str());
        return nullptr;
    }

    // Build flat rule pointer array for init
    std::vector<const flm_grammar_element *> rule_ptrs;
    rule_ptrs.reserve(parser.rules.size());
    for (const auto & r : parser.rules) rule_ptrs.push_back(r.data());

    const size_t n_rules         = rule_ptrs.size();
    const size_t start_rule_idx  = parser.symbol_ids.at(grammar_root);

    // Copy rules into owned vectors
    flm_grammar_rules vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        const flm_grammar_element * p = rule_ptrs[i];
        for (; p->type != FLM_GRETYPE_END; p++) vec_rules[i].push_back(*p);
        vec_rules[i].push_back({FLM_GRETYPE_END, 0});
    }

    // Check for left recursion
    std::vector<bool> visited(n_rules, false);
    std::vector<bool> in_progress(n_rules, false);
    std::vector<bool> may_be_empty(n_rules, false);
    for (size_t i = 0; i < n_rules; i++) {
        if (visited[i]) continue;
        if (detect_left_recursion(vec_rules, i, visited, in_progress, may_be_empty)) {
            fprintf(stderr, "flm_grammar: left recursion at rule %zu\n", i);
            return nullptr;
        }
    }

    // Build initial stacks from the start rule
    flm_grammar_stacks stacks;
    const flm_grammar_element * pos = vec_rules[start_rule_idx].data();
    do {
        flm_grammar_stack stack;
        if (!is_end_of_sequence(pos)) stack.push_back(pos);
        advance_stack(vec_rules, stack, stacks);
        while (!is_end_of_sequence(pos)) pos++;
        if (pos->type == FLM_GRETYPE_ALT) pos++;
        else break;
    } while (true);

    auto g = std::unique_ptr<FlmGrammar>(new FlmGrammar());
    g->rules_        = std::move(vec_rules);
    g->stacks_       = std::move(stacks);
    g->partial_utf8_ = {};
    g->token_pieces_  = token_pieces;
    g->eog_ids_       = eog_ids;
    return g;
}

void FlmGrammar::apply(float * logits, int vocab_size) const {
    bool allow_eog = false;
    for (const auto & stack : stacks_) {
        if (stack.empty()) { allow_eog = true; break; }
    }

    // Decode every token to UTF-8 codepoints, build candidate list
    std::vector<std::pair<std::vector<uint32_t>, flm_partial_utf8>> decoded;
    decoded.reserve(vocab_size);
    flm_grammar_candidates candidates;
    candidates.reserve(vocab_size);

    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] == -std::numeric_limits<float>::infinity()) continue;

        // Check EOG
        bool is_eog = false;
        for (int eid : eog_ids_) {
            if (i == eid) { is_eog = true; break; }
        }
        if (is_eog) {
            if (!allow_eog) logits[i] = -std::numeric_limits<float>::infinity();
            continue;
        }

        if (i >= static_cast<int>(token_pieces_.size())) {
            logits[i] = -std::numeric_limits<float>::infinity();
            continue;
        }

        const std::string & piece = token_pieces_[i];
        if (piece.empty() || piece[0] == 0) {
            logits[i] = -std::numeric_limits<float>::infinity();
            continue;
        }

        decoded.push_back(decode_utf8(piece, partial_utf8_));
        candidates.push_back({
            static_cast<size_t>(i),
            decoded.back().first.data(),
            i,  // token_id for token-level matching
            decoded.back().second
        });
    }

    auto rejects = reject_candidates(rules_, stacks_, candidates);
    for (const auto & r : rejects) {
        logits[r.index] = -std::numeric_limits<float>::infinity();
    }
}

void FlmGrammar::accept(int token_id) {
    if (token_id < 0 || token_id >= static_cast<int>(token_pieces_.size())) return;

    // Check EOG
    for (int eid : eog_ids_) {
        if (token_id == eid) {
            // EOG token — verify grammar allows ending
            for (const auto & stack : stacks_) {
                if (stack.empty()) return;
            }
            return;  // best-effort: don't crash if grammar doesn't allow end yet
        }
    }

    const std::string & piece = token_pieces_[token_id];
    if (piece.empty()) return;

    auto decoded = decode_utf8(piece, partial_utf8_);
    const auto & cps = decoded.first;

    flm_grammar_stacks new_stacks;
    new_stacks.reserve(stacks_.size());

    for (const auto & stack : stacks_) {
        if (stack.empty()) continue;

        const flm_grammar_element * pos = stack.back();

        // Token-level matching: if top of stack is a token rule, match by ID
        if (pos->type == FLM_GRETYPE_TOKEN || pos->type == FLM_GRETYPE_TOKEN_NOT) {
            if (grammar_match_token(pos, token_id)) {
                flm_grammar_stack ns(stack.begin(), stack.end() - 1);
                if (!is_end_of_sequence(pos + 1)) ns.push_back(pos + 1);
                advance_stack(rules_, ns, new_stacks);
            }
            continue;
        }

        // Character-level matching
        flm_grammar_stacks current = {stack};
        for (auto it = cps.begin(), end = cps.end() - 1; it != end; ++it) {
            flm_grammar_stacks next;
            for (const auto & s : current) {
                if (s.empty()) continue;
                const flm_grammar_element * p = s.back();
                if (p->type == FLM_GRETYPE_TOKEN || p->type == FLM_GRETYPE_TOKEN_NOT)
                    continue;  // skip token rules during char processing
                auto match = grammar_match_char(p, *it);
                if (match.first) {
                    flm_grammar_stack ns(s.begin(), s.end() - 1);
                    if (!is_end_of_sequence(match.second)) ns.push_back(match.second);
                    advance_stack(rules_, ns, next);
                }
            }
            current = std::move(next);
            if (current.empty()) break;
        }

        for (auto & s : current) {
            if (std::find(new_stacks.begin(), new_stacks.end(), s) == new_stacks.end())
                new_stacks.emplace_back(std::move(s));
        }
    }

    stacks_       = std::move(new_stacks);
    partial_utf8_ = decoded.second;
}

std::unique_ptr<FlmGrammar> FlmGrammar::clone() const {
    auto g = std::unique_ptr<FlmGrammar>(new FlmGrammar());
    g->rules_        = rules_;
    g->stacks_       = stacks_;
    g->partial_utf8_ = partial_utf8_;
    g->token_pieces_  = token_pieces_;
    g->eog_ids_       = eog_ids_;

    // Fix up stack pointers to point into the cloned rules
    for (size_t is = 0; is < g->stacks_.size(); is++) {
        for (size_t ie = 0; ie < g->stacks_[is].size(); ie++) {
            for (size_t ir = 0; ir < rules_.size(); ir++) {
                for (size_t je = 0; je < rules_[ir].size(); je++) {
                    if (stacks_[is][ie] == &rules_[ir][je]) {
                        g->stacks_[is][ie] = &g->rules_[ir][je];
                    }
                }
            }
        }
    }
    return g;
}
