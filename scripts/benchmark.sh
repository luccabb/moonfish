#!/bin/bash
#
# Local Stockfish benchmark script
# Replicates the CI benchmark workflow for local testing
#

set -euo pipefail

# Configuration
ROUNDS=${ROUNDS:-20}
CONCURRENCY=${CONCURRENCY:-4}
SKILL_LEVELS=${SKILL_LEVELS:-"1 3 4 5"}
MOONFISH_TIME=${MOONFISH_TIME:-60}  # seconds per move
STOCKFISH_TC=${STOCKFISH_TC:-"60+5"}
OUTPUT_DIR=${OUTPUT_DIR:-"benchmark_results"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Moonfish vs Stockfish Benchmark ==="
echo ""

# Check dependencies
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        echo "Please install $1 and try again"
        exit 1
    fi
}

echo "Checking dependencies..."
check_dependency cutechess-cli
check_dependency stockfish

# Check for moonfish binary
MOONFISH_BIN="./dist/moonfish"
if [ ! -f "$MOONFISH_BIN" ]; then
    echo -e "${YELLOW}Moonfish binary not found at $MOONFISH_BIN${NC}"
    echo "Building moonfish..."
    make build-lichess
fi

if [ ! -f "$MOONFISH_BIN" ]; then
    echo -e "${RED}Error: Could not find or build moonfish binary${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Rounds per skill level: $ROUNDS"
echo "  Concurrency: $CONCURRENCY"
echo "  Skill levels: $SKILL_LEVELS"
echo "  Moonfish time: ${MOONFISH_TIME}s per move"
echo "  Stockfish time control: $STOCKFISH_TC"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run benchmarks for each skill level
for SKILL in $SKILL_LEVELS; do
    echo -e "${YELLOW}=== Running benchmark vs Stockfish Skill Level $SKILL ===${NC}"

    PGN_FILE="$OUTPUT_DIR/benchmark-skill$SKILL.pgn"
    LOG_FILE="$OUTPUT_DIR/benchmark-skill$SKILL.log"

    cutechess-cli \
        -engine name=moonfish cmd="$MOONFISH_BIN" dir=. proto=uci tc=inf st="$MOONFISH_TIME" timemargin=10000 \
        -engine name=stockfish cmd=stockfish proto=uci option.Skill\ Level="$SKILL" option.Threads=1 tc="$STOCKFISH_TC" timemargin=10000 \
        -rounds "$ROUNDS" \
        -repeat \
        -concurrency "$CONCURRENCY" \
        -pgnout "$PGN_FILE" \
        -recover \
        2>&1 | tee "$LOG_FILE"

    echo ""
done

# Parse and display results
echo ""
echo -e "${GREEN}=== RESULTS ===${NC}"
echo ""

for SKILL in $SKILL_LEVELS; do
    LOG_FILE="$OUTPUT_DIR/benchmark-skill$SKILL.log"
    PGN_FILE="$OUTPUT_DIR/benchmark-skill$SKILL.pgn"

    if [ -f "$LOG_FILE" ]; then
        echo "### vs Stockfish Skill Level $SKILL"

        # Extract score
        SCORE=$(grep "Score of moonfish vs stockfish:" "$LOG_FILE" | tail -1)
        if [ -n "$SCORE" ]; then
            echo "$SCORE"
        fi

        # Parse win/loss/draw
        WINS=$(echo "$SCORE" | sed -E 's/.*: ([0-9]+) - ([0-9]+) - ([0-9]+).*/\1/' || echo 0)
        LOSSES=$(echo "$SCORE" | sed -E 's/.*: ([0-9]+) - ([0-9]+) - ([0-9]+).*/\2/' || echo 0)
        DRAWS=$(echo "$SCORE" | sed -E 's/.*: ([0-9]+) - ([0-9]+) - ([0-9]+).*/\3/' || echo 0)
        TOTAL=$((WINS + LOSSES + DRAWS))

        if [ "$TOTAL" -gt 0 ]; then
            WIN_RATE=$(echo "scale=1; $WINS * 100 / $TOTAL" | bc)
            echo "Win rate: ${WIN_RATE}%"
        fi

        # Non-checkmate endings
        if [ -f "$PGN_FILE" ]; then
            ENDINGS=$(grep -oE ', [^}]+\}' "$PGN_FILE" | sed 's/, //; s/}//' | grep -v 'mates' | sort | uniq -c | sort -rn 2>/dev/null || true)
            if [ -n "$ENDINGS" ]; then
                echo "Non-checkmate endings:"
                echo "$ENDINGS" | while read count ending; do
                    echo "  - $ending: $count"
                done
            fi
        fi

        echo ""
    fi
done

echo "Results saved to $OUTPUT_DIR/"
echo "  - PGN files: benchmark-skill*.pgn"
echo "  - Log files: benchmark-skill*.log"
