"""
==============================================================================
playbook.py
==============================================================================

Playbook state management for ACE (Agentic Context Engineering) implementation.

This module provides the Playbook class for:
- Parsing playbook content with sections and bullets
- Applying curator operations (ADD, UPDATE, REMOVE)
- Deduplication for Pathway i (parallel processing)
- ID generation for new bullets

Compatible with FrozenLake prompts and ACE-AppWorld patterns.

Playbook Format:
    ## SECTION HEADER
    [prefix-NNNNN] Bullet content here
    [prefix-NNNNN] Another bullet content

    ## ANOTHER SECTION
    [prefix-NNNNN] More content

Example:
    ## STRATEGIES AND HARD RULES
    [shr-00001] Before your first move, identify the Goal (G) position.
    [shr-00002] Before each move, check if the target tile contains a Hole (O).

    ## COMMON MISTAKES
    [cm-00001] Moving directly toward the goal without first scanning for holes.

Supported Operations:
    - ADD: Add new bullet to a section
    - UPDATE: Modify existing bullet content by ID
    - REMOVE: Delete bullet by ID
"""

import re
from typing import Any

from .utils import (
    get_section_slug,
    normalize_section_name,
)


class Playbook:
    """
    Manages playbook state with sections and bullets.

    For Pathway i (parallel processing): Includes deduplication at apply time
    to mitigate redundancy from parallel Curator calls.

    Attributes:
        content: Raw playbook text
        next_global_id: Next ID number for new bullets
        sections: Known section definitions with prefixes
    """

    # Known sections with their slug prefixes
    # Maps normalized section name -> {"header": display header, "prefix": slug}
    KNOWN_SECTIONS = {
        # FrozenLake sections
        "strategies_and_hard_rules": {"header": "## STRATEGIES AND HARD RULES", "prefix": "shr"},
        "common_mistakes": {"header": "## COMMON MISTAKES", "prefix": "cm"},
        "grid_patterns": {"header": "## GRID PATTERNS", "prefix": "gp"},
        # Fallback section
        "others": {"header": "## OTHERS", "prefix": "misc"},
    }

    def __init__(self, initial_content: str = "", flat_mode: bool = None):
        """
        Initialize Playbook with content.

        Args:
            initial_content: Initial playbook text (can be empty)
            flat_mode: If True, use flat structure (no sections, integer IDs,
                      auto-renumber). If False, use section-based structure.
                      If None (default), auto-detect from content:
                      content with ## headers → section mode,
                      empty or no headers → flat mode.
        """
        self.content = initial_content.strip() if initial_content else ""
        if flat_mode is None:
            flat_mode = not bool(re.search(r"^##\s", self.content, re.MULTILINE))
        self.flat_mode = flat_mode
        self.next_global_id = self._find_max_id() + 1

    # =========================================================================
    # Core Methods
    # =========================================================================

    def to_string(self) -> str:
        """
        Return current playbook as string.

        Used for:
        - Creating snapshots for parallel processing (Pathway i)
        - Injecting into Generator prompts
        - Saving to disk

        Returns:
            Current playbook content
        """
        return self.content

    def apply_operations(
        self,
        operations: dict,
    ) -> int:
        """
        Apply Curator operations to playbook.

        Supports:
        - ADD: Add new bullet to specified section
        - UPDATE: Modify existing bullet content by ID
        - REMOVE: Delete bullet by ID

        Args:
            operations: Dict with "operations" list from Curator
                       Each operation has format:
                       - ADD: {"type": "ADD", "section": "...", "content": "..."}
                       - UPDATE: {"type": "UPDATE", "bullet_id": "...", "content": "..."}
                       - REMOVE: {"type": "REMOVE", "bullet_id": "..."}

        Returns:
            Number of operations successfully applied
        """
        if not operations:
            return 0

        # Handle both dict format and direct list
        if isinstance(operations, dict):
            ops_list = operations.get("operations", [])
        elif isinstance(operations, list):
            ops_list = operations
        else:
            return 0

        if not ops_list:
            return 0

        applied_count = 0

        for op in ops_list:
            if not isinstance(op, dict):
                continue

            op_type = op.get("type", "").upper()

            if op_type == "ADD":
                content = op.get("content", "").strip()
                section = op.get("section", "others")

                if not content:
                    continue

                # Add the bullet
                success = self._add_bullet(section, content)
                if success:
                    applied_count += 1

            elif op_type == "UPDATE":
                bullet_id = op.get("bullet_id", "").strip()
                content = op.get("content", "").strip()

                if not bullet_id or not content:
                    continue

                # Update the bullet
                success = self._update_bullet(bullet_id, content)
                if success:
                    applied_count += 1

            elif op_type == "REMOVE":
                bullet_id = op.get("bullet_id", "").strip()

                if not bullet_id:
                    continue

                # Remove the bullet
                success = self._remove_bullet(bullet_id)
                if success:
                    applied_count += 1

        # Auto-renumber in flat mode to keep IDs sequential
        if self.flat_mode and applied_count > 0:
            self._renumber_bullets()

        return applied_count

    def apply_operations_with_report(
        self,
        operations: dict | list,
        *,
        normalize_tip_id: bool = True,
    ) -> dict[str, Any]:
        """Apply the explicit ``feedback_v1`` mutation contract.

        ``apply_operations`` remains the legacy contract. This opt-in path
        accepts the old ``tip_id`` alias, validates each operation, and counts
        only actual content changes. IDs refer to the original batch until
        the final flat-book renumbering, just as in the legacy implementation.
        The JSON-safe report records canonical operations and why a requested
        change did not apply. Conflicting aliases never select a target.
        """
        before = self.content
        report: dict[str, Any] = {
            "contract": "canonical_operations_v1",
            "requested_count": 0,
            "applied_count": 0,
            "changed": False,
            "normalized_tip_id_count": 0,
            "operations": [],
        }
        ops_list = operations.get("operations") if isinstance(operations, dict) else operations
        if not isinstance(ops_list, list):
            report["error"] = "operations_must_be_a_list"
            return report
        report["requested_count"] = len(ops_list)

        def clean_id(value: Any) -> str | None:
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                return None
            value = str(value).strip().strip("[]").strip()
            return value or None

        for index, op in enumerate(ops_list):
            item: dict[str, Any] = {"index": index, "status": "invalid"}
            report["operations"].append(item)
            if not isinstance(op, dict):
                item["reason"] = "operation_must_be_an_object"
                continue
            op_type = op.get("type")
            op_type = op_type.upper().strip() if isinstance(op_type, str) else ""
            item["type"] = op_type
            if op_type not in {"ADD", "UPDATE", "REMOVE"}:
                item["reason"] = "unknown_operation_type"
                continue

            canonical: dict[str, Any] = {"type": op_type}
            if op_type in {"UPDATE", "REMOVE"}:
                bullet_id = clean_id(op.get("bullet_id"))
                alias_id = clean_id(op.get("tip_id")) if normalize_tip_id else None
                if "bullet_id" in op and bullet_id is None:
                    item["reason"] = "invalid_bullet_id"
                    continue
                if normalize_tip_id and "tip_id" in op and alias_id is None:
                    item["reason"] = "invalid_tip_id"
                    continue
                if bullet_id is not None and alias_id is not None and bullet_id != alias_id:
                    item["reason"] = "conflicting_bullet_id_and_tip_id"
                    continue
                if bullet_id is None and alias_id is not None:
                    bullet_id = alias_id
                    item["normalized_tip_id"] = True
                    report["normalized_tip_id_count"] += 1
                if bullet_id is None:
                    item["reason"] = "missing_bullet_id"
                    continue
                canonical["bullet_id"] = bullet_id

            if op_type in {"ADD", "UPDATE"}:
                content = op.get("content")
                if not isinstance(content, str) or not content.strip():
                    item["reason"] = "missing_or_invalid_content"
                    continue
                canonical["content"] = content.strip()
            if op_type == "ADD":
                section = op.get("section", "others")
                if not isinstance(section, str):
                    item["reason"] = "invalid_section"
                    continue
                canonical["section"] = section
            item["canonical_operation"] = canonical

            previous = self.content
            if op_type == "ADD":
                success = self._add_bullet(canonical["section"], canonical["content"])
            elif op_type == "UPDATE":
                success = self._update_bullet(canonical["bullet_id"], canonical["content"])
            else:
                success = self._remove_bullet(canonical["bullet_id"])
            if not success:
                item["status"] = "not_applied"
                item["reason"] = "bullet_not_found" if op_type != "ADD" else "add_failed"
            elif self.content == previous:
                item["status"] = "unchanged"
                item["reason"] = "content_unchanged"
            else:
                item["status"] = "applied"
                report["applied_count"] += 1

        if self.flat_mode and report["applied_count"]:
            self._renumber_bullets()
        report["changed"] = self.content != before
        return report

    def _extract_all_bullets(self) -> list[str]:
        """
        Extract all bullet contents from playbook.

        Returns:
            List of bullet content strings (without IDs)
        """
        bullets = []

        for line in self.content.split("\n"):
            parsed = parse_playbook_line(line)
            if parsed:
                bullets.append(parsed["content"])

        return bullets

    # =========================================================================
    # Bullet Operations
    # =========================================================================

    def _add_bullet(self, section: str, content: str) -> bool:
        """
        Add a new bullet to the specified section.

        In flat_mode: ignores section, appends with integer ID.
        In section mode: adds to the specified section with prefix ID.

        Args:
            section: Section name (ignored in flat_mode)
            content: Bullet content (without ID)

        Returns:
            True if successfully added
        """
        if self.flat_mode:
            # Flat mode: simple integer ID, append to end
            new_id = str(self.next_global_id)
            self.next_global_id += 1
            new_line = format_playbook_line(new_id, content)

            if self.content and not self.content.endswith("\n"):
                self.content += "\n"
            self.content += new_line
            print(f"  ✅ Added bullet [{new_id}]")
            return True

        # Section mode: use prefix-based IDs and section headers
        # Normalize section name
        normalized_section = normalize_section_name(section)

        # Get section info
        section_info = self.KNOWN_SECTIONS.get(normalized_section, self.KNOWN_SECTIONS.get("others"))

        if not section_info:
            # Fallback to generating slug
            prefix = get_section_slug(normalized_section)
        else:
            prefix = section_info["prefix"]

        # Generate new ID
        new_id = f"{prefix}-{self.next_global_id:05d}"
        self.next_global_id += 1

        # Format bullet line
        new_line = format_playbook_line(new_id, content)

        # Find section and insert
        inserted = self._insert_bullet_into_section(normalized_section, new_line)

        if inserted:
            print(f"  ✅ Added bullet [{new_id}] to section '{normalized_section}'")
            return True
        else:
            # Section not found - append to end or create section
            self._append_bullet(normalized_section, new_line)
            print(f"  ✅ Added bullet [{new_id}] (appended to '{normalized_section}')")
            return True

    def _update_bullet(self, bullet_id: str, new_content: str) -> bool:
        """
        Update an existing bullet's content by ID.

        The bullet ID is preserved; only the content is replaced.

        Args:
            bullet_id: Bullet ID (with or without brackets)
            new_content: New content for the bullet

        Returns:
            True if bullet was found and updated
        """
        clean_id = bullet_id.strip("[]")
        lines = self.content.split("\n")
        updated = False

        for i, line in enumerate(lines):
            parsed = parse_playbook_line(line)
            if parsed and parsed["id"] == clean_id:
                # Found the bullet - replace with new content
                new_line = format_playbook_line(clean_id, new_content)
                lines[i] = new_line
                updated = True
                print(f"  ✏️  Updated bullet [{clean_id}]")
                break

        if updated:
            self.content = "\n".join(lines)
        else:
            print(f"  ⚠️  Bullet [{clean_id}] not found for UPDATE")

        return updated

    def _remove_bullet(self, bullet_id: str) -> bool:
        """
        Remove a bullet by its ID.

        Args:
            bullet_id: Bullet ID (with or without brackets)

        Returns:
            True if bullet was found and removed
        """
        clean_id = bullet_id.strip("[]")
        lines = self.content.split("\n")
        new_lines = []
        removed = False

        for line in lines:
            parsed = parse_playbook_line(line)
            if parsed and parsed["id"] == clean_id:
                removed = True
                print(f"  🗑️  Removed bullet [{clean_id}]")
                continue  # Skip this line
            new_lines.append(line)

        if removed:
            self.content = "\n".join(new_lines)
        else:
            print(f"  ⚠️  Bullet [{clean_id}] not found for REMOVE")

        return removed

    def _insert_bullet_into_section(self, section: str, bullet_line: str) -> bool:
        """
        Insert bullet line into the appropriate section.

        FIX: Now correctly inserts BEFORE trailing blank lines, preserving
             proper formatting with blank lines between sections.

        Args:
            section: Normalized section name
            bullet_line: Formatted bullet line with ID

        Returns:
            True if section found and bullet inserted
        """
        lines = self.content.split("\n")

        # Find section
        section_idx = -1
        next_section_idx = len(lines)

        for i, line in enumerate(lines):
            if line.strip().startswith("##"):
                header_normalized = normalize_section_name(line.strip()[2:])
                if header_normalized == section:
                    section_idx = i
                elif section_idx >= 0:
                    # Found next section after target
                    next_section_idx = i
                    break

        if section_idx < 0:
            return False

        # =====================================================================
        # FIX: Back up past trailing blank lines before inserting
        # =====================================================================
        # Find the correct insertion point: BEFORE any blank lines that
        # separate this section from the next section header.
        #
        # BEFORE FIX:
        #   [shr-00004] ...
        #                    ← blank line
        #   [shr-00005] ...  ← inserted here (WRONG - after blank)
        #   ## NEXT SECTION
        #
        # AFTER FIX:
        #   [shr-00004] ...
        #   [shr-00005] ...  ← inserted here (CORRECT - before blank)
        #                    ← blank line preserved
        #   ## NEXT SECTION
        # =====================================================================

        insert_idx = next_section_idx

        # Back up past trailing blank lines
        while insert_idx > section_idx + 1 and not lines[insert_idx - 1].strip():
            insert_idx -= 1

        # Insert at the correct position (with blank line separator)
        if insert_idx > section_idx + 1 and lines[insert_idx - 1].strip():
            lines.insert(insert_idx, "")
            insert_idx += 1
        lines.insert(insert_idx, bullet_line)
        self.content = "\n".join(lines)
        return True

    def _append_bullet(self, section: str, bullet_line: str):
        """
        Append bullet to playbook, creating section if needed.

        Args:
            section: Normalized section name
            bullet_line: Formatted bullet line with ID
        """
        section_header = self._get_section_header(section)

        # Check if section exists
        if section_header.lower() not in self.content.lower():
            # Create section at end
            if self.content and not self.content.endswith("\n"):
                self.content += "\n"
            self.content += f"\n{section_header}\n"

        # Find where to append in existing section
        lines = self.content.split("\n")
        section_idx = -1

        for i, line in enumerate(lines):
            if line.strip().startswith("##"):
                header_normalized = normalize_section_name(line.strip()[2:])
                if header_normalized == section:
                    section_idx = i
                elif section_idx >= 0:
                    # Found next section - insert before it
                    # Back up past blank lines
                    insert_idx = i
                    while insert_idx > section_idx + 1 and not lines[insert_idx - 1].strip():
                        insert_idx -= 1
                    # Add blank line separator before new bullet
                    if insert_idx > section_idx + 1 and lines[insert_idx - 1].strip():
                        lines.insert(insert_idx, "")
                        insert_idx += 1
                    lines.insert(insert_idx, bullet_line)
                    self.content = "\n".join(lines)
                    return

        # No next section found - append at end
        if self.content and not self.content.endswith("\n"):
            self.content += "\n"
        self.content += "\n" + bullet_line + "\n"

    # =========================================================================
    # ID Management
    # =========================================================================

    def _renumber_bullets(self):
        """
        Renumber all bullets sequentially (flat_mode only).

        After ADD/REMOVE operations, IDs may have gaps.
        This renumbers all bullets as [1], [2], [3], ...
        and resets next_global_id accordingly.
        """
        lines = self.content.split("\n")
        new_lines = []
        counter = 1

        for line in lines:
            parsed = parse_playbook_line(line)
            if parsed:
                new_lines.append(format_playbook_line(str(counter), parsed["content"]))
                counter += 1
            else:
                new_lines.append(line)

        self.content = "\n".join(new_lines)
        self.next_global_id = counter

    def _find_max_id(self) -> int:
        """
        Find the maximum ID number in the playbook.

        Handles both formats:
        - Flat mode: [1], [2], [3] → extracts integer directly
        - Section mode: [shr-00001] → extracts number after dash

        Returns:
            Highest ID number found (0 if none)
        """
        max_id = 0

        for line in self.content.split("\n"):
            parsed = parse_playbook_line(line)
            if parsed:
                # Try section-style ID first (prefix-NNNNN)
                id_match = re.search(r"-(\d+)$", parsed["id"])
                if id_match:
                    num = int(id_match.group(1))
                    max_id = max(max_id, num)
                else:
                    # Try flat-style ID (pure integer)
                    try:
                        num = int(parsed["id"])
                        max_id = max(max_id, num)
                    except ValueError:
                        pass

        return max_id

    # =========================================================================
    # Section Management
    # =========================================================================

    def _get_section_header(self, section: str) -> str:
        """
        Get the header line for a section.

        Args:
            section: Normalized section name

        Returns:
            Section header string (e.g., "## STRATEGIES AND HARD RULES")
        """
        section_info = self.KNOWN_SECTIONS.get(section)
        if section_info:
            return section_info["header"]

        # Generate header from section name
        header = section.replace("_", " ").upper()
        return f"## {header}"

    def get_sections(self) -> list[str]:
        """
        Get list of sections in the playbook.

        Returns:
            List of normalized section names
        """
        sections = []

        for line in self.content.split("\n"):
            if line.strip().startswith("##"):
                section = normalize_section_name(line.strip()[2:])
                sections.append(section)

        return sections

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict:
        """
        Get statistics about the playbook.

        Returns:
            Dict with total_bullets, by_section counts
        """
        stats = {"total_bullets": 0, "by_section": {}}

        current_section = "general"

        for line in self.content.split("\n"):
            if line.strip().startswith("##"):
                current_section = normalize_section_name(line.strip()[2:])
                continue

            parsed = parse_playbook_line(line)
            if parsed:
                stats["total_bullets"] += 1

                if current_section not in stats["by_section"]:
                    stats["by_section"][current_section] = 0
                stats["by_section"][current_section] += 1

        return stats

    def get_bullet_by_id(self, bullet_id: str) -> dict | None:
        """
        Get a bullet by its ID.

        Args:
            bullet_id: Bullet ID (with or without brackets)

        Returns:
            Parsed bullet dict or None
        """
        # Normalize ID
        clean_id = bullet_id.strip("[]")

        for line in self.content.split("\n"):
            parsed = parse_playbook_line(line)
            if parsed and parsed["id"] == clean_id:
                return parsed

        return None

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, filepath: str):
        """
        Save playbook to file.

        Args:
            filepath: Path to save file
        """
        with open(filepath, "w") as f:
            f.write(self.content)
        print(f"💾 Saved playbook to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "Playbook":
        """
        Load playbook from file.

        Args:
            filepath: Path to load from

        Returns:
            New Playbook instance
        """
        with open(filepath) as f:
            content = f.read()
        return cls(content)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"Playbook({stats['total_bullets']} bullets, {len(stats['by_section'])} sections)"

    def __str__(self) -> str:
        """Return full playbook text for string coercion."""
        return self.to_string()


# =============================================================================
# Module-Level Functions (for compatibility with ACE-AppWorld)
# =============================================================================


def parse_playbook_line(line: str) -> dict | None:
    """
    Parse a single playbook line to extract components.

    Supports format
    1) "[id] content" (simple format)

    Args:
        line: Single line from playbook

    Returns:
        Dict with {id, content, raw_line} or None

    Examples:
        >>> parse_playbook_line("[shr-00001] Check for holes before moving")
        {'id': 'shr-00001', 'content': 'Check for holes before moving', 'raw_line': '[shr-00001] Check for holes before moving'}
    """
    text = line.strip()

    if not text or text.startswith("##"):
        return None

    # Simple format (FrozenLake)
    # [id] content
    pattern_simple = r"\[([^\]]+)\]\s*(.*)"
    match2 = re.match(pattern_simple, text)
    if match2:
        return {"id": match2.group(1), "content": match2.group(2).strip(), "raw_line": line}

    return None


def format_playbook_line(bullet_id: str, content: str) -> str:
    """
    Format a bullet into playbook line format.

    Args:
        bullet_id: Bullet ID (e.g., "shr-00001")
        content: Bullet content

    Returns:
        Formatted line: "[shr-00001] content"
    """
    # Ensure ID is clean (no brackets)
    clean_id = bullet_id.strip("[]")
    return f"[{clean_id}] {content}"


def get_next_global_id(playbook_text: str) -> int:
    """
    Extract highest global ID from playbook and return next one.

    Args:
        playbook_text: Full playbook text

    Returns:
        Next available ID number
    """
    max_id = 0

    for line in playbook_text.strip().split("\n"):
        parsed = parse_playbook_line(line)
        if parsed:
            id_match = re.search(r"-(\d+)$", parsed["id"])
            if id_match:
                num = int(id_match.group(1))
                max_id = max(max_id, num)

    return max_id + 1


def apply_curator_operations(playbook_text: str, operations: list[dict], next_id: int) -> tuple[str, int]:
    """
    Apply curator operations to playbook text.

    Functional interface for compatibility with ACE-AppWorld.

    Args:
        playbook_text: Current playbook text
        operations: List of operation dicts
        next_id: Next available ID number

    Returns:
        Tuple of (updated_playbook_text, new_next_id)
    """
    playbook = Playbook(playbook_text)
    playbook.next_global_id = next_id

    playbook.apply_operations({"operations": operations})

    return playbook.to_string(), playbook.next_global_id


def extract_playbook_bullets_by_ids(playbook_text: str, bullet_ids: list[str]) -> str:
    """
    Extract specific bullets from playbook by their IDs.

    Used by Reflector to show which bullets were used.

    Args:
        playbook_text: Full playbook text
        bullet_ids: List of bullet IDs to extract

    Returns:
        Formatted string with requested bullets
    """
    if not bullet_ids:
        return "(No bullets referenced)"

    playbook = Playbook(playbook_text)
    found_bullets = []

    for bid in bullet_ids:
        bullet = playbook.get_bullet_by_id(bid)
        if bullet:
            found_bullets.append(f"[{bullet['id']}] {bullet['content']}")

    if not found_bullets:
        return "(Referenced bullets not found in playbook)"

    return "\n".join(found_bullets)
