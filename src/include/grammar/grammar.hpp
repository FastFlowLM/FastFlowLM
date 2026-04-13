/// \file grammar.hpp
/// \brief GBNF grammar-constrained sampling for FastFlowLM
/// \author FastFlowLM Community (ported from llama.cpp)
/// \date 2026-04-12
/// \note GBNF grammar parser and pushdown-automaton enforcer, adapted from
///       llama.cpp's llama-grammar.h / llama-grammar.cpp (MIT-licensed).
///       Strips lazy grammars, trigger patterns, and vocab-dependent token
///       parsing (<token> syntax) to keep the port minimal and self-contained.
///       Only character-level matching and numeric token IDs (<[id]>) are
///       supported.
#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

// ---- grammar element types (mirrors llama.cpp) ----------------------------

enum flm_gretype {
    FLM_GRETYPE_END            = 0,  // end of rule definition
    FLM_GRETYPE_ALT            = 1,  // start of alternate definition
    FLM_GRETYPE_RULE_REF       = 2,  // non-terminal: reference to rule
    FLM_GRETYPE_CHAR           = 3,  // terminal: character (code point)
    FLM_GRETYPE_CHAR_NOT       = 4,  // inverse char(s)
    FLM_GRETYPE_CHAR_RNG_UPPER = 5,  // inclusive range upper bound
    FLM_GRETYPE_CHAR_ALT       = 6,  // alternate char in set
    FLM_GRETYPE_CHAR_ANY       = 7,  // any character (.)
    FLM_GRETYPE_TOKEN          = 8,  // terminal: token (<[id]>)
    FLM_GRETYPE_TOKEN_NOT      = 9,  // inverse token (!<[id]>)
};

struct flm_grammar_element {
    flm_gretype type;
    uint32_t    value;  // Unicode code point or rule ID
};

// ---- partial UTF-8 state --------------------------------------------------

struct flm_partial_utf8 {
    uint32_t value    = 0;   // accumulated bits (unshifted)
    int      n_remain = 0;   // bytes remaining; -1 = invalid
};

// ---- candidate for grammar filtering --------------------------------------

struct flm_grammar_candidate {
    size_t             index;        // position in the logits array
    const uint32_t   * code_points;  // decoded UTF-8 codepoints (0-terminated)
    int                token_id;     // token ID for token-level matching
    flm_partial_utf8   partial_utf8;
};

// ---- type aliases ---------------------------------------------------------

using flm_grammar_rule   = std::vector<flm_grammar_element>;
using flm_grammar_stack  = std::vector<const flm_grammar_element *>;
using flm_grammar_rules  = std::vector<flm_grammar_rule>;
using flm_grammar_stacks = std::vector<flm_grammar_stack>;
using flm_grammar_candidates = std::vector<flm_grammar_candidate>;

// ---- GBNF parser ----------------------------------------------------------

struct flm_grammar_parser {
    std::map<std::string, uint32_t> symbol_ids;
    flm_grammar_rules rules;

    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);
    void     add_rule(uint32_t rule_id, const flm_grammar_rule & rule);

    const char * parse_alternates(const char * src, const std::string & rule_name,
                                  uint32_t rule_id, bool is_nested);
    const char * parse_sequence(const char * src, const std::string & rule_name,
                                flm_grammar_rule & rule, bool is_nested);
    const char * parse_rule(const char * src);
    bool parse(const char * src);
};

// ---- grammar state (pushdown automaton) -----------------------------------

class FlmGrammar {
public:
    /// Create from a GBNF grammar string.
    /// \param grammar_str   GBNF text
    /// \param grammar_root  name of the start rule (default "root")
    /// \param token_pieces   vector mapping token-id → decoded text
    /// \param eog_ids        end-of-generation token ids
    /// \return nullptr on parse error
    static std::unique_ptr<FlmGrammar> create(
        const std::string & grammar_str,
        const std::string & grammar_root,
        const std::vector<std::string> & token_pieces,
        const std::vector<int> & eog_ids);

    /// Apply grammar constraints: set logits[i] = -inf for every token that
    /// the grammar rejects in the current state.  Call BEFORE top-k / softmax.
    void apply(float * logits, int vocab_size) const;

    /// Accept a sampled token and advance the grammar state.
    /// Call AFTER sampling.
    void accept(int token_id);

    /// Deep-clone the grammar (including internal stack state).
    std::unique_ptr<FlmGrammar> clone() const;

private:
    flm_grammar_rules  rules_;
    flm_grammar_stacks stacks_;
    flm_partial_utf8   partial_utf8_;

    std::vector<std::string> token_pieces_;  // token-id → text
    std::vector<int>         eog_ids_;

    // internal helpers
    static void advance_stack(const flm_grammar_rules & rules,
                              const flm_grammar_stack & stack,
                              flm_grammar_stacks & new_stacks);

    static flm_grammar_candidates reject_candidates(
        const flm_grammar_rules & rules,
        const flm_grammar_stacks & stacks,
        const flm_grammar_candidates & candidates);

    static flm_grammar_candidates reject_candidates_for_stack(
        const flm_grammar_rules & rules,
        const flm_grammar_stack & stack,
        const flm_grammar_candidates & candidates);
};
