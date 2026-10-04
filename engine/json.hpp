// json.hpp — a minimal JSON reader, enough for Track-3 scenario.json / batch.json.
//
// Hand-rolled rather than vendored because the submission image must be self-contained and
// dependency-free, and the schema surface here is small and fully enumerated:
//   objects, arrays, strings, numbers (int + float + exponent), true / false / null.
// No unicode escapes beyond the simple two-character ones, because the scenarios contain none.
//
// Numbers are kept as a double PLUS the original integer when the text had no '.'/'e', so that
// nanosecond values like 1612483200000000000 survive exactly -- a double would quantise them.

#ifndef JSON_HPP
#define JSON_HPP

#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace minijson {

class Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

enum class Kind { Null, Bool, Int, Double, String, Array, Object };

class Value {
public:
    Kind kind = Kind::Null;
    bool b = false;
    std::int64_t i = 0;
    double d = 0.0;
    std::string s;
    Array arr;
    Object obj;

    bool is_null() const { return kind == Kind::Null; }
    bool has(const std::string& k) const {
        return kind == Kind::Object && obj.find(k) != obj.end();
    }
    const Value& at(const std::string& k) const {
        auto it = obj.find(k);
        if (it == obj.end()) throw std::runtime_error("json: missing key '" + k + "'");
        return it->second;
    }

    // Accessors with defaults -- scenario fields are frequently optional.
    std::int64_t as_int(std::int64_t dflt = 0) const {
        if (kind == Kind::Int) return i;
        if (kind == Kind::Double) return static_cast<std::int64_t>(d);
        if (kind == Kind::Bool) return b ? 1 : 0;
        return dflt;
    }
    double as_double(double dflt = 0.0) const {
        if (kind == Kind::Double) return d;
        if (kind == Kind::Int) return static_cast<double>(i);
        return dflt;
    }
    bool as_bool(bool dflt = false) const {
        if (kind == Kind::Bool) return b;
        if (kind == Kind::Int) return i != 0;
        return dflt;
    }
    std::string as_string(const std::string& dflt = "") const {
        return kind == Kind::String ? s : dflt;
    }

    std::int64_t get_int(const std::string& k, std::int64_t dflt) const {
        return has(k) ? at(k).as_int(dflt) : dflt;
    }
    double get_double(const std::string& k, double dflt) const {
        return has(k) ? at(k).as_double(dflt) : dflt;
    }
    bool get_bool(const std::string& k, bool dflt) const {
        return has(k) ? at(k).as_bool(dflt) : dflt;
    }
    std::string get_string(const std::string& k, const std::string& dflt) const {
        return has(k) ? at(k).as_string(dflt) : dflt;
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : t_(text) {}

    Value parse() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        return v;
    }

private:
    [[noreturn]] void fail(const std::string& why) const {
        throw std::runtime_error("json: " + why + " at offset " + std::to_string(p_));
    }
    void skip_ws() {
        while (p_ < t_.size()) {
            const char c = t_[p_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++p_;
            else break;
        }
    }
    char peek() const {
        if (p_ >= t_.size()) fail("unexpected end of input");
        return t_[p_];
    }
    void expect(char c) {
        if (p_ >= t_.size() || t_[p_] != c) fail(std::string("expected '") + c + "'");
        ++p_;
    }

    Value parse_value() {
        switch (peek()) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': {
                Value v;
                v.kind = Kind::String;
                v.s = parse_string();
                return v;
            }
            case 't': case 'f': return parse_bool();
            case 'n': return parse_null();
            default: return parse_number();
        }
    }

    Value parse_object() {
        Value v;
        v.kind = Kind::Object;
        expect('{');
        skip_ws();
        if (peek() == '}') { ++p_; return v; }
        while (true) {
            skip_ws();
            const std::string key = parse_string();
            skip_ws();
            expect(':');
            skip_ws();
            v.obj[key] = parse_value();
            skip_ws();
            if (peek() == ',') { ++p_; continue; }
            expect('}');
            return v;
        }
    }

    Value parse_array() {
        Value v;
        v.kind = Kind::Array;
        expect('[');
        skip_ws();
        if (peek() == ']') { ++p_; return v; }
        while (true) {
            skip_ws();
            v.arr.push_back(parse_value());
            skip_ws();
            if (peek() == ',') { ++p_; continue; }
            expect(']');
            return v;
        }
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (p_ >= t_.size()) fail("unterminated string");
            const char c = t_[p_++];
            if (c == '"') return out;
            if (c != '\\') { out.push_back(c); continue; }
            if (p_ >= t_.size()) fail("unterminated escape");
            const char e = t_[p_++];
            switch (e) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    // Scenario descriptions contain \u2014 (em-dash), so non-ASCII codepoints
                    // must be encoded as UTF-8 rather than refused. Surrogate pairs are joined.
                    if (p_ + 4 > t_.size()) fail("short \\u escape");
                    unsigned long cp =
                        std::strtoul(t_.substr(p_, 4).c_str(), nullptr, 16);
                    p_ += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && p_ + 6 <= t_.size() &&
                        t_[p_] == '\\' && t_[p_ + 1] == 'u') {
                        const unsigned long lo =
                            std::strtoul(t_.substr(p_ + 2, 4).c_str(), nullptr, 16);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            p_ += 6;
                        }
                    }
                    if (cp < 0x80) {
                        out.push_back(static_cast<char>(cp));
                    } else if (cp < 0x800) {
                        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    } else if (cp < 0x10000) {
                        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    } else {
                        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    }
                    break;
                }
                default: fail("bad escape");
            }
        }
    }

    Value parse_bool() {
        Value v;
        v.kind = Kind::Bool;
        if (t_.compare(p_, 4, "true") == 0) { v.b = true; p_ += 4; return v; }
        if (t_.compare(p_, 5, "false") == 0) { v.b = false; p_ += 5; return v; }
        fail("bad literal");
    }

    Value parse_null() {
        if (t_.compare(p_, 4, "null") != 0) fail("bad literal");
        p_ += 4;
        return Value{};
    }

    // Integers keep full int64 precision; only values with '.' or an exponent become doubles.
    Value parse_number() {
        const std::size_t start = p_;
        if (p_ < t_.size() && (t_[p_] == '-' || t_[p_] == '+')) ++p_;
        bool is_float = false;
        while (p_ < t_.size()) {
            const char c = t_[p_];
            if (c >= '0' && c <= '9') { ++p_; continue; }
            if (c == '.' || c == 'e' || c == 'E') { is_float = true; ++p_; continue; }
            if ((c == '-' || c == '+') && (t_[p_ - 1] == 'e' || t_[p_ - 1] == 'E')) { ++p_; continue; }
            break;
        }
        if (p_ == start) fail("bad number");
        const std::string num = t_.substr(start, p_ - start);
        Value v;
        if (is_float) {
            v.kind = Kind::Double;
            v.d = std::strtod(num.c_str(), nullptr);
        } else {
            v.kind = Kind::Int;
            v.i = std::strtoll(num.c_str(), nullptr, 10);
            v.d = static_cast<double>(v.i);
        }
        return v;
    }

    const std::string& t_;
    std::size_t p_ = 0;
};

inline Value parse(const std::string& text) { return Parser(text).parse(); }

}  // namespace minijson

#endif  // JSON_HPP
